import json
import logging
import os
import requests
import shutil
import tarfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict
from PIL import Image
import yaml
from tqdm import tqdm
from loguru import logger as loguru_logger
import numpy as np
import torch
import time


CODE_SUFFIXES = {".py", ".sh", ".yaml", ".yml"}  # Python codes  # Shell scripts  # Configuration files


def safe_dir(path):
    """
    Create a directory (or the parent directory of a file) if it does not exist.

    Args:
        path (str or Path): Path to the directory.

    Returns:
        path (Path): Path object of the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def safe_save_file(save_path, *args, save_fn=None, **kwargs):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_save_path = save_path.parent / f"temp_{save_path.name}"
    save_to = None
    try:
        save_to = save_fn(tmp_save_path, *args, **kwargs)
        shutil.copyfile(tmp_save_path, save_path)
        save_to = save_path
        tmp_save_path.unlink()
    except Exception as e:
        print(f"Failed to save to {save_path}. {type(e)}: {e}")
    return save_to


def safe_json(data):
    """
    Make sure the input data is serializable to JSON.
    """
    try:
        json.dumps(data)
    except TypeError as e:
        print(f"Failed to serialize data to JSON. {e.__class__.__name__}: {e}")
    return data


def is_valid_experiment(path):
    path = Path(path)
    if path.is_dir() and path.name.split("_")[0].isdigit():
        return True
    return False


def get_experiment_max_number(experiments):
    valid_experiment_numbers = []
    for exp in experiments:
        if is_valid_experiment(exp):
            valid_experiment_numbers.append(int(Path(exp).name.split("_")[0]))
    if valid_experiment_numbers:
        return max(valid_experiment_numbers)
    return 0


def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def rank0_logger(rank):
    if rank == 0:
        from loguru import logger
    else:
        logger = empty_logger()
    return logger


def logger_filter(name):
    def filter_(record):
        return record["extra"].get("name") == name

    return filter_


def log_string_for_torch_tensor(tensor, name):
    sig_digits = 6
    # Start with cheap operations
    log_string = f"{name} \t shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}"
    
    # Optional: Add a size threshold for expensive operations
    if tensor.numel() > 256*16:  # Skip for tensors larger than 10M elements
        log_string += " (range omit)"
        return log_string
        
    # Do expensive operations only if needed
    log_string += f", range: {tensor.min():.{sig_digits}g}, {tensor.max():.{sig_digits}g}"
    log_string += f", first element: {tensor.flatten()[0]:.{sig_digits}g}"
    
    # Most expensive operations last, and skip the bfloat16 conversion
    tensor = tensor.to(torch.float32)
    log_string += f", mean: {tensor.mean():.{sig_digits}g}, std: {tensor.std():.{sig_digits}g}"
    
    return log_string


def log_string_for_float_torch_tensor_dict(variable, name):
    if isinstance(variable, dict):
        log_string = f"{name}: Keys of input dict: {variable.keys()}\n"
        for k, v in variable.items():
            k_string = log_string_for_float_torch_tensor_dict(v, k)
            log_string += k_string + "\n"
    elif isinstance(variable, (int, float, np.generic)):
        log_string = f"{name}: type: {type(variable)}, value: {variable:.3f}"
    elif isinstance(variable, torch.Tensor)and variable.dim() == 0:
        # Handle single-element tensors
        log_string = f"{name}: type: {variable.dtype}, value: {variable.item():.3f}"
    elif isinstance(variable, (torch.Tensor, torch.nn.parameter.Parameter)):
        log_string = log_string_for_torch_tensor(variable, name)
    elif isinstance(variable, np.ndarray):
        log_string = log_string_for_torch_tensor(torch.from_numpy(variable), name)
    elif isinstance(variable, Image.Image):
        log_string = f"{name}: type: {type(variable)}, value: {variable}"
        numpy_variable = np.array(variable)
        log_string = log_string_for_float_torch_tensor_dict(numpy_variable, name)
        # print(log_string)
    else:
        log_string = f"{name}: {variable}"
        
        # raise ValueError(f"Unsupported type: {type(variable)}")
    return log_string


def log_in_safe_logger(variable="", logger="loguru", name="", should_log=True):
    if not should_log or logger is None or os.environ.get("DEBUG", "false") == "false":
        return None
    else:
        log_string = log_string_for_float_torch_tensor_dict(variable, name)
        if logger == "print":
            print(log_string)
        elif logger == "loguru":
            loguru_logger.info(log_string)
        else:
            logger.info(log_string)
        return log_string


def safe_check_grad(name, tensor, logger):
    if logger is None:
        from loguru import logger
    if tensor.requires_grad:
        # print(f"Registering hook for {name}")
        # from loguru import logger
        logger.info(f"Registering hook for {name}")
        @torch.no_grad()
        def gradient_hook_func(grad):
            log_in_safe_logger(grad, logger, f"Gradient for {name}: ")
            torch.cuda.empty_cache()
            return grad
        tensor.register_hook(gradient_hook_func)
    else:
        # from loguru import logger
        logger.info(f"No gradient for {name}")
        # print(f"No gradient for {name}")
    return tensor


def dict_repr(obj):
    return '\n    ' + yaml.safe_dump(obj, indent=4).replace('\n', '\n    ')


def recursive_vars(obj):
    obj_dict = vars(obj)
    for key, value in obj_dict.items():
        if hasattr(value, '__dict__'):
            obj_dict[key] = recursive_vars(value)
        if isinstance(value, list):
            obj_dict[key] = [recursive_vars(item) if hasattr(item, '__dict__') else item for item in value]
    return obj_dict


def recursive_keys(args):
    keys = set()
    if isinstance(args, list):
        for item in args:
            keys.update(recursive_keys(item))
    elif isinstance(args, dict):
        for key, value in args.items():
            keys.add(key)
            keys.update(recursive_keys(value))
    return keys


def remove_dup_keys(args):
    if hasattr(args, '__dict__'):
        args = recursive_vars(args)
    all_sub_keys = set()
    for key, value in args.items():
        all_sub_keys.update(recursive_keys(value))
    dedup_args = {key: value for key, value in args.items() if key not in all_sub_keys}
    return dedup_args


def dump_configs(args, save_path, extra_args=None) -> Dict:
    # args_dict = remove_dup_keys(args)
    from copy import deepcopy

    args = deepcopy(args)
    if hasattr(args, '__dict__'):
        args_dict = recursive_vars(args)
    else:
        args_dict = args

    if extra_args:
        assert isinstance(extra_args, dict), f"extra_args should be a dictionary, got {type(extra_args)}."
        args_dict.update(extra_args)
    # Save to file
    with safe_file(Path(save_path).with_suffix('.yaml')).open("w") as f:
        # Add a root key to align with user defined config files
        yaml.dump({'all_configs': args_dict}, f, indent=4)
    # with safe_file(Path(save_path).with_suffix('.json')).open('w') as f:
    #     json.dump(args_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    return args_dict


def dump_codes(save_path, root, sub_dirs=None, valid_suffixes=None, save_prefix="./"):
    """
    Dump codes to the experiment directory.

    Args:
        save_path (str or Path): Path to the experiment directory.
        root (Path): Path to the root directory of the codes.
        sub_dirs (list): List of subdirectories to be dumped. If None, all files in the root directory will
            be dumped. (default: None)
        valid_suffixes (tuple, optional): Valid suffixes of the files to be dumped. If None, CODE_SUFFIXES will be used.
            (default: None)
        save_prefix (str, optional): Prefix to be added to the files in the tarball. (default: './')
    """
    if valid_suffixes is None:
        valid_suffixes = CODE_SUFFIXES

    # Force to use tar.gz suffix
    save_path = safe_file(save_path)
    assert save_path.name.endswith(".tar.gz"), f"save_path should end with .tar.gz, got {save_path.name}."
    # Make root absolute
    root = Path(root).absolute()
    # Make a tarball of the codes
    with tarfile.open(save_path, "w:gz") as tar:
        # Recursively add all files in the root directory
        if sub_dirs is None:
            sub_dirs = list(root.iterdir())
        for sub_dir in sub_dirs:
            for file in Path(sub_dir).rglob("*"):
                if file.is_file() and file.suffix in valid_suffixes:
                    # make file absolute
                    file = file.absolute()
                    arcname = Path(save_prefix) / file.relative_to(root)
                    tar.add(file, arcname=arcname)
    return root


def resolve_resume_path(resume, results_dir):
    # Detect the resume path. Support both the experiment index and the full path.
    if resume.isnumeric():
        tmp_dirs = list(Path(results_dir).glob("*"))
        id2exp_dir = defaultdict(list)
        for tmp_dir in tmp_dirs:
            part0 = tmp_dir.name.split("_")[0]
            if part0.isnumeric():
                id2exp_dir[int(part0)].append(tmp_dir)
        resume_id = int(resume)
        valid_exp_dir = id2exp_dir.get(resume_id)
        if len(valid_exp_dir) == 0:
            raise ValueError(
                f"No valid experiment directories found in {results_dir} with the experiment " f"index {resume}."
            )
        elif len(valid_exp_dir) > 1:
            raise ValueError(
                f"Multiple valid experiment directories found in {results_dir} with the experiment "
                f"index {resume}: {valid_exp_dir}."
            )
        resume_path = valid_exp_dir[0] / "checkpoints"
    else:
        resume_path = Path(resume)

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume path {resume_path} not found.")

    return resume_path


def get_next_available_save_id(src_dir):
    src_dir = Path(src_dir)
    existed_files = list(src_dir.glob("*"))
    valid_ids = []
    for existed_files in existed_files:
        head = existed_files.name.split("_")[0]
        if head.isdigit():
            valid_ids.append(int(head))
    if not valid_ids:
        return 0
    return max(valid_ids) + 1


def read_file_to_bytesio(file_path, chunk_size=1024 * 1024 * 64):
    file_size = os.path.getsize(file_path)
    buffer = BytesIO()
    with open(file_path, 'rb') as f:
        pbar = tqdm(range(0, file_size, chunk_size), desc="Reading", unit="B", unit_scale=True, total=file_size)
        for _ in pbar:
            byte_s = f.read(chunk_size)
            buffer.write(byte_s)
            pbar.update(chunk_size)
    buffer.seek(0)
    return buffer


def download(url, dst):
    dst = Path(dst)
    if not dst.exists():
        print(f"Downloading {url} to {dst}")
        response = requests.get(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as f:
            f.write(response.content)
        print(f"Downloaded.")


def extract(src, save_dir):
    dst_dir = Path(save_dir)
    if not (dst_dir / Path(src).name.split('.')[0]).exists():
        print(f"Extracting {src} to {save_dir}")
        with tarfile.open(src, "r:gz") as tar:
            tar.extractall(save_dir)
        print(f"Extracted.")


def download_and_extract(url, save_dir, file_name=None):
    if file_name is None:
        # If file_name is None, try to get the file name from the url
        if '?' in url:
            file_name = url.split('?')[0].split('/')[-1]
        else:
            file_name = url.split('/')[-1]
    tar_path = Path(save_dir) / file_name
    for _ in range(30):
        try:
            # 纯下载无法判断是否正确，因为cos下载失败也会生成一个文件
            download(url, tar_path)
            # 只有下载成功才会正确解压出来
            extract(tar_path, save_dir)
            break
        except Exception as e:
            print(f"Download failed: {e.__class__.__name__}: {e}. Retrying...")
            time.sleep(2)
    else:
        raise RuntimeError(f"Failed to download {url} after 10 attempts.")


def save_to_csv(dataframe, save_path, append=False):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if append:
        dataframe.to_csv(save_path, index=False, mode='a', header=not save_path.exists())
    else:
        dataframe.to_csv(save_path, index=False)


def save_to_json(save_path, results, indent=4):
    with open(save_path, "w") as f:
        json.dump(results, f, indent=indent)
    return save_path
