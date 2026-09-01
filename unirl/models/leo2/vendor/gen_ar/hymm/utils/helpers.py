import collections.abc
from itertools import repeat
import importlib
import yaml
import re
import fnmatch


def count_zh_en_words(text):
    # 中文字符数
    zh_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 英文单词数
    en_words = re.findall(r'[A-Za-z]+', text)
    en_count = len(en_words)
    # 总长度（可选：按字符数或按“中文字符+英文单词”总数）
    total = zh_count + en_count
    zh_ratio = zh_count / total if total else 0
    en_ratio = en_count / total if total else 0
    return {
        'zh_count': zh_count,
        'en_count': en_count,
        'zh_ratio': zh_ratio,
        'en_ratio': en_ratio
    }


def default(value, default_val):
    return default_val if value is None else value


def default_dtype(value, default_val):
    if value is not None:
        assert isinstance(value, type(default_val)), f"Expect {type(default_val)}, got {type(value)}."
        return value
    return default_val


def repeat_interleave(lst, num_repeats):
    return [item for item in lst for _ in range(num_repeats)]


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def as_tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def as_list_of_2tuple(x):
    x = as_tuple(x)
    if len(x) == 1:
        x = (x[0], x[0])
    assert len(x) % 2 == 0, f"Expect even length, got {len(x)}."
    lst = []
    for i in range(0, len(x), 2):
        lst.append((x[i], x[i + 1]))
    return lst


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n - (n % k) + k


def merge_dicts(dict1, dict2):
    """递归合并两个字典"""
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            # 如果两个字典都有这个键，并且都是字典，则递归合并
            merge_dicts(dict1[key], value)
        else:
            # 否则，直接覆盖
            dict1[key] = value
    return dict1


def merge_yaml_files(file_list):
    merged_config = {}

    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config:
                # Remove the first level
                for key, value in config.items():
                    if isinstance(value, dict):
                        merged_config = merge_dicts(merged_config, value)  # 使用递归合并字典
                    else:
                        merged_config[key] = value

    return merged_config


def merge_dict(file_list):
    merged_config = {}

    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config:
                merged_config = merge_dicts(merged_config, config)  # 使用递归合并字典

    return merged_config


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def readable_time(src, key=None, left_steps=None):
    """ Convert time seconds to a readable format: DD Days, HH Hours, MM Minutes, SS Seconds """
    if hasattr(src, "timers"):
        assert key is not None, "key must be provided when src is a Timer object"
        time_repr = (
            f"Average: {readable_time(src.average(key))}"
            f" | Elapsed: {readable_time(src.elapsed_total(key))}"
        )
        if left_steps is not None:
            assert isinstance(left_steps, int), "left_steps must be int"
            time_repr += f" | Remain: {readable_time(src.average(key) * left_steps)}"
        return time_repr

    seconds = int(src)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f"{days} Days, {hours} Hours, {minutes} Minutes, {seconds} Seconds"
    if hours > 0:
        return f"{hours} Hours, {minutes} Minutes, {seconds} Seconds"
    if minutes > 0:
        return f"{minutes} Minutes, {seconds} Seconds"
    return f"{seconds} Seconds"


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def multi_pattern_match(name, patterns):
    """Check if a name matches any of the given patterns.

    Args:
        name (str): The name to be checked.
        patterns (list or str): A list of patterns or a single pattern string.
            Patterns can include wildcards like '*' and '?'.

    Returns:
        bool: True if the name matches any of the patterns, False otherwise.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def print_args(title, args):
    print(f'------------------------ {title} ------------------------', flush=True)
    str_list = []
    name_max_length = max([48] + [len(arg) + 3 for arg in vars(args)])
    for arg in vars(args):
        dots = '.' * (name_max_length - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print(f'-------------------- end of {title} ---------------------', flush=True)
