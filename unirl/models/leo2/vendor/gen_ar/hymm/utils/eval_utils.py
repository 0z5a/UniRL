import torch
import pandas as pd
import numpy as np


def cn_len(s):
    length = 0
    for char in s:
        if "\u4e00" <= char <= "\u9fff":  # 中文字符的 unicode 范围
            length += 2
        else:
            length += 1
    return length


def format_s(s, max_width):
    """
    Format string to specified max width. Chinese characters are counted as 2 characters.
    """
    if cn_len(s) <= max_width:
        return s

    start, end = 0, len(s)
    count_length_left = 0
    count_length_right = 0
    flag = True
    while count_length_left + count_length_right < max_width - 3:
        if flag:
            count_length_left += 2 if "\u4e00" <= s[start] <= "\u9fff" else 1
            start += 1
        else:
            count_length_right += 2 if "\u4e00" <= s[end - 1] <= "\u9fff" else 1
            end -= 1
        flag = not flag

    return s[:start] + "..." + s[end:]


def stylize_dataframe(df, max_width):
    """
    Stylize dataframe according to specified column widths.

    Parameters:
    df: pandas DataFrame
    max_width: dictionary mapping column names to desired max widths

    Returns:
    df: stylized DataFrame
    """
    df = df.copy()
    df.set_index("id", inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(str).apply(lambda x: format_s(x, max_width))

    return df


def batch_data_repr(batch, max_width=100):
    """
    Format batch to specified max width. Chinese characters are counted as 2 characters.

    Parameters:
    batch: dictionary mapping keys to values
    max_width: int

    Returns:
    batch: list of formatted strings
    """
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.tolist()
        else:
            new_batch[k] = v
    df = pd.DataFrame(new_batch)
    df = stylize_dataframe(df, max_width)
    return df
