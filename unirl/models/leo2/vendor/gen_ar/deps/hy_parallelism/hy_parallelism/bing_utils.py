"""
Note(kevinkhwu): This file contains miscellaneous utility functions and scripts intended for my personal, day-to-day use.
These are not part of the core release package of hy_parallelism. If you are unsure how to use any function here, it is recommended not to use it.

Author: kevinkhwu
Email: kevinkhwu@tencent.com
"""

import os
import time
import shutil
import inspect
import zipfile
import builtins
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Union, Optional, Tuple

import einops

import loguru
from loguru import logger

from torch import nn
from torch import distributed as dist


config_loguru = False
if config_loguru:
    from .common.logging import get_logger
    logger = get_logger(__name__)
    # logger.remove(None)
    # import os
    # # print(os.environ)
    # import loguru, sys
    # def logfunc(x):
    #     return os.environ.get('LOCAL_RANK', '0') == '0'
    # level = 'DEBUG'
    # loguru.logger.remove(None)
    # rank = os.environ.get('RANK', '0')
    # loguru.logger.add(sys.stdout, level=level, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:^8}</level> | <level><bold>[Rank "+ rank + "]: {message}</bold></level> (<cyan>{file}:{line}</cyan>)",)
    # loguru.logger.level('INFO', color="<green>")
    # loguru.logger.level('DEBUG', color="<fg 100,100,100>")

# Ensure loguru is available
try:
    import loguru
except ImportError:
    # Fallback to basic logging if loguru is not available
    import logging
    logging.basicConfig(level=logging.INFO)
    loguru = logging.getLogger(__name__)


def thread_run(func):
    from threading import Thread
    Thread(target=func).start()

def is_local(path):
    return False
    path = Path(path).resolve().absolute().as_posix()
    return path.startswith('/apdcephfs_nj8/') or path.startswith('/root') or path.startswith('/tmp')


def gather_obj(obj, group=None):
    if group is None:
        ws = dist.get_world_size()
    else:
        ws = dist.get_world_size(group)
    lst = [None for _ in range(ws)]
    dist.all_gather_object(lst, obj, group=group)
    return lst

__temp_file_list = []
_file_mapping_path = '/tmp/file_mapping'

def update_file_mapping(new_dic):
    import json
    if Path(_file_mapping_path).exists():
        try:
            dic = json.loads(open(_file_mapping_path, 'r').read())
        except:
            dic = {}
    else:
        dic = {}
    dic.update(new_dic)
    with open(_file_mapping_path, 'w') as f:
        json.dump(dic, f)

def get_tmp_mapped_path(path):
    import json
    if Path(_file_mapping_path).exists():
        try:
            dic = json.loads(open(_file_mapping_path, 'r').read())
            ret_path = dic.get(path, None)
            if ret_path and Path(ret_path).exists():
                return ret_path
        except:
            ...
    return None

def get_nonlocal_file(path, delete=False):
    if not (isinstance(path, str) or isinstance(path, os.PathLike)):
        return path
    path = Path(path).resolve().absolute().as_posix()
    import tempfile, loguru
    from torch import distributed as dist
    # if dist.get_world_size() > 8:
    #     return get_nonlocal_file_new(path)

    if tmp_mapped_path := get_tmp_mapped_path(path):
        loguru.logger.info(f'从mapping file {_file_mapping_path},  {path} 映射到 {tmp_mapped_path}')
        return tmp_mapped_path
    # else:
    #     possible = []
    #     for f in Path('/tmp').glob('**/*'):
    #         if f.name == Path(path).name:
    #             possible.append(f)
    #     if possible:
    #         possible.sort(key=lambda x: x.stat().st_mtime)
    #         loguru.logger.info('临时文件还没删，直接用最新的')
    #         ret = str(possible[-1])
    #         update_file_mapping({path: ret})
    #         return ret

    if is_local(path):
        return path
    local_rank = os.environ.get('LOCAL_RANK', '0')
    if not Path(path).exists():
        raise FileNotFoundError(f'{path} not found')
    if local_rank == '0':
        import inspect
        temp_dir_args = inspect.signature(tempfile.TemporaryDirectory).parameters
        if 'delete' in temp_dir_args:
            temp = tempfile.TemporaryDirectory(delete=delete)
        else:
            temp = tempfile.TemporaryDirectory()
        __temp_file_list.append(temp)
        loguru.logger.info(f'Copying non local file {path} to {temp.name}')
        # if Path(path).is_dir():
        #     shutil.copytree(path, temp.name)
        # else:
        import subprocess
        proc = subprocess.Popen(['rsync', '-aP', '--copy-links', path, temp.name])
        proc.wait()
        loguru.logger.info(f'Copy done')
        local_file_path = Path(temp.name) / Path(path).name
        local_file_path = local_file_path.as_posix()
        if not delete:
            update_file_mapping({path: local_file_path})
    if dist.is_initialized():
        if local_rank == '0':
            all_gather_obj = local_file_path
        else:
            all_gather_obj = None
        gather_list = gather_obj(all_gather_obj)
        # print(gather_list)
        local_file_path = gather_list[get_rank() // 8 * 8]
        assert Path(local_file_path).exists(), f'{get_rank()} 不存在'
    else:
        # raise RuntimeError('Initialize dist env first before calling get_nonloacal file')
        assert local_rank == '0'
    return local_file_path

def get_nonlocal_file_new(path):
    from torch import distributed as dist
    if not dist.is_initialized() or dist.get_world_size() < 8:
        return get_nonlocal_file(path)
    if is_local(path):
        return path
    assert dist.is_initialized()
    import tempfile, loguru
    from torch import distributed as dist
    local_rank = os.environ.get('LOCAL_RANK', '0')
    global_rank = os.environ.get('RANK', '0')
    if global_rank == '0':
        temp = tempfile.TemporaryDirectory()
        __temp_file_list.append(temp)
        loguru.logger.info(f'Copying non local file {path} to {temp.name}')
        # if Path(path).is_dir():
        #     shutil.copytree(path, temp.name)
        # else:
        import subprocess
        proc = subprocess.Popen(['rsync', '-aP', path, temp.name])
        proc.wait()
        loguru.logger.info(f'Copy done')
        local_file_path = Path(temp.name) / Path(path).name
        local_file_path = local_file_path.as_posix()
    master_path = get_synced_object(lambda:local_file_path, 0)
    if int(global_rank) > 7:
        # assert not Path(master_path).exists()
        if local_rank == '0':
            master_ip = os.environ['MASTER_ADDR']
            import subprocess
            temp = tempfile.TemporaryDirectory()
            __temp_file_list.append(temp)
            cmd = ['rsync', '-aP',
                   # '-e', 'ssh -P 36000',
                   f'root@{master_ip}:{master_path}', temp.name]
            retrieved_path = (Path(temp.name) / Path(path).name).as_posix()
            loguru.logger.info(f'从主节点复制: {cmd}')
            proc = subprocess.Popen(cmd)
            proc.wait()
        else:
            retrieved_path = ''
    else:
        retrieved_path = ''
    ret = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(ret, retrieved_path)
    if int(global_rank) < 7:
        return master_path
    else:
        if local_rank == 0:
            return retrieved_path
        else:
            return ret[int(global_rank) // 8 * 8]


def copy_by_suffix(src, dst, suffixes, replace=False, excludes=[]):
    if replace:
        params = ['rsync', '-aPuivv', '--progress', f'{src}/', dst]
    else:
        params = ['rsync', '-aPuivv', '--progress', f'{src}', dst]
    extra_param = ['--max-size', '2m']
    for ex in excludes:
        extra_param += ["--exclude", ex + '/']
    extra_param += ["--include", "*/", ]
    for suffix in suffixes:
        while suffix.startswith('.'):
            suffix = suffix[1:]
        extra_param += ["--include", f"*.{suffix}"]
    extra_param += ["--exclude", "*"]
    params += extra_param
    loguru.logger.debug(params)
    loguru.logger.debug(' '.join(params))
    import subprocess as sp
    proc = sp.Popen(params, stdout=sp.DEVNULL)
    proc.wait()


def pack_py_files(source_folder, target_folder):
    # source: 项目根目录
    # target_folder: 一个文件夹，最后会生成在  target/xxxx.zip


    # 生成zip文件名
    current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    zip_file_name = f"{current_time}.zip"
    zip_file_path = os.path.join(target_folder, zip_file_name)

    # 创建临时文件夹用于保存源文件夹的目录结构
    temp_folder = os.path.join(target_folder, f"{current_time}_temp")
    os.makedirs(temp_folder, exist_ok=True)

    try:
        # 复制源文件夹的目录结构到临时文件夹

        suffix = ['py', 'ipynb', 'cc', 'cpp', 'c', 'java', 'zsh', 'bash',  'sh', 'md', 'conf', 'xml', 'json', '.txt', 'yaml',
                  ]
        loguru.logger.info('Copying code by suffixes')
        copy_by_suffix(source_folder, temp_folder, suffixes=suffix, excludes=['output', 'outputs', 'compare', 'evaluate_cache', 'evaluate_cache.old', 'debug', 'results', 'results.old'])
        with open(os.path.join(temp_folder, 'run.sh'), 'w') as f:
            f.write(f'# {repr(sys.argv)}\n')
            def command_list_to_zsh_str(cmd_args):
                import shlex
                return shlex.join(cmd_args)
            for k, v in os.environ.items():
                f.write(f'export {k}={v}\n')
            f.write(f'python3 {command_list_to_zsh_str(sys.argv)}\n')
        # shutil.copytree(source_folder, temp_folder)

        # 遍历临时文件夹，将.py文件添加到zip文件中
        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_folder):
                for file in files:
                    # if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    file_size_mb = Path(file_path).stat().st_size / 1024. / 1024.
                    # loguru.logger.info(f'{file}: {file_size_mb}')
                    if file_size_mb > 2:
                        loguru.logger.warning(f'Ignore large file {file_path} with size {file_size_mb:.2f} MB')
                        continue
                    arcname = os.path.relpath(file_path, os.path.join(temp_folder, Path(source_folder).name))
                    arcname = os.path.join(f'{current_time}', arcname)
                    # print(file_path, arcname)
                    zipf.write(file_path, arcname)
            # zipf.write(temp_folder, f'{current_time}')

        loguru.logger.debug(f"成功打包成 {zip_file_path}")
    except Exception as e:
        loguru.logger.error(f"打包过程中出现错误: ")
        loguru.logger.exception(e)
    finally:
        # 清理临时文件夹
        shutil.rmtree(temp_folder)


def unnormalize_frames(normalized_frames): # b c t h w [-1, 1] -> [0, 255] b t h w c, numpy
    # assert normalized_frames.min() < 0
    assert normalized_frames.min() > -2 and normalized_frames.max() < 2 # 一定在 -1 到 1 之间，适当放宽点 assert, for numeric stability
    frames = einops.rearrange((normalized_frames + 1) * 127.5, 'b c t h w -> b t h w c')
    if not isinstance(frames, np.ndarray):
        frames = frames.detach().cpu().numpy()
    frames = frames.clip(min=0, max=255)
    return frames.astype(np.uint8)
    # else:
    #     frames = frames.clamp(min=0, max=255)
    #     return frames.type(torch.uint8)


def assert_shape(x, shape_pattern):
    parse = einops.parse_shape(x, shape_pattern)
    valid_c = [3, 4]


    assert parse['h'] % 8 == 0
    assert parse['w'] % 8 == 0
    assert parse['c'] in valid_c
    # if 't' in shape_pattern:
    #     assert parse['t'] in [global_config.num_frames, 1, global_config.n_frame_cond, 16], str(parse)

    return


    valid_h = [448, 640, 512, 256, 64, 32]
    valid_w = [768, 1024, 768, 640, 512, 256, 64, 32]
    for foo in valid_h.copy():
        valid_h.append(foo//8)
    for foo in valid_w.copy():
        valid_w.append(foo//8)
    assert parse['h'] in valid_h
    assert parse['w'] in valid_w
    assert parse['c'] in valid_c

def normalize_frames(frames, is_image=None):
    ret = _normalize_frames(frames, is_image)
    assert_shape(ret, 'b c t h w')
    return ret

def _normalize_frames(frames, is_image=None):
    from PIL import Image
    if isinstance(frames, Image.Image):
        return _normalize_frames(np.array(frames), is_image)
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames).contiguous()
    elif isinstance(frames, list):
        assert len(frames) > 0
        if isinstance(frames[0], np.ndarray):
            frames = np.stack(frames)
            frames = torch.from_numpy(frames)
        elif isinstance(frames[0], Image.Image):
            frames = [np.array(f) for f in frames]
            return _normalize_frames(frames, is_image)
        elif isinstance(frames[0], torch.Tensor):
            frames = torch.stack(frames)
        else:
            raise ValueError(f'Unsupported frame type {type(frames)}')
    if frames.max() > 5:
        # unnormalized data,
        # Could be one of     b h w c,   t h w c,  b t h w c
        # Could be one of        t h w c
        frames = frames / 127.5 - 1
        if len(frames.shape) == 3:
            shape = 'h w c'
            # assert frames.shape[0] == 256
            assert_shape(frames, shape)
            return einops.rearrange(frames, f'{shape} -> 1 c 1 h w')
        elif len(frames.shape) == 4:
            # assert frames.shape[1] == 256
            assert is_image is not None, f'Ambiguous shape {frames.shape} requires `is_image` argument'
            if is_image:
                shape = 'b h w c'
                assert_shape(frames, shape)
                return einops.rearrange(frames, f'{shape} -> b c 1 h w')
            else:
                shape = 't h w c'
                assert_shape(frames, shape)
                return einops.rearrange(frames, 't h w c -> 1 c t h w')
        elif len(frames.shape) == 5:
            # assert frames.shape[2] == 256
            shape = 'b t h w c'
            assert_shape(frames, shape)
            return einops.rearrange(frames, f'{shape} -> b c t h w')
        else:
            raise ValueError(f'Invalid unnormalized input with shape {frames.shape}.')
    else:
        def handle_shape():
            if len(frames.shape) == 3:
                # should never happen, wierd data shape....  (Normalized but without the batch dimension..)
                raise Exception('Should never happen, wierd data shape....  (Normalized but without the batch dimension..)')
                # assert frames.shape[0] == 256
                return einops.rearrange(frames, 'c h w -> 1 c 1 h w')
            elif len(frames.shape) == 4:
                # b h w c
                # t h w c
                # assert frames.shape[-1] == 256
                assert is_image is not None, f'Ambiguous shape {frames.shape} requires `is_image` argument'
                if is_image:
                    shape = 'b c h w'
                    assert_shape(frames, shape)
                    return einops.rearrange(frames, f'{shape} -> b c 1 h w')
                else:
                    shape = 'c t h w'
                    assert_shape(frames, shape)
                    return einops.rearrange(frames, f'{shape} -> 1 c t h w')
            elif len(frames.shape) == 5:
                # assert frames.shape[-1] == 256
                shape = 'b c t h w'
                assert_shape(frames, shape)
                return frames
            else:
                raise ValueError(f'Invalid unnormalized input with shape {frames.shape}.')

        def handle_range(vid):
            # 全落在 [0,1], 大概率还没归一化, 当然也有可能某些特殊数据归一化后还在0,1, 比如纯白图片，对这种图片再归一化一次也无伤大雅
            if (vid.min() > 0 or is_close(vid.min(), 0)) and (vid.max() < 1 or is_close(vid.max(), 1)):
                return (vid - 0.5) / 0.5
            # 存在负数值了，大概率已经归一化了, 当然也有可能由于精度误差，未归一化的也存在负值，但通常不会有 1e-3 这么大
            elif vid.min() < 0 - 1e-3:
                return vid
            else:
                raise Exception(f'Unknown data range min:{vid.min()}, max:{vid.max()}, mean:{vid.mean()}, std:{vid.std()}')
        ret = handle_shape()
        return handle_range(ret)

def to_tensor(frames, is_image=None): # 返回 -1, 1    而不是 [0, 1]
    return normalize_frames(frames, is_image)

def to_video(frames, is_image=None):
    return unnormalize_frames(normalize_frames(frames, is_image))

def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list):
        if len(x) > 0:
            if isinstance(x[0], np.ndarray):
                return to_torch(np.stack(x))
            else:
                return torch.stack(x)
        else:
            return torch.tensor([])
    else:
        raise ValueError(f'Unsupported type {type(x)}')

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        if len(x) > 0:
            if isinstance(x[0], np.ndarray):
                return np.stack(x)
            else:
                return to_numpy(torch.stack(x))
        else:
            return np.array([])
    else:
        raise ValueError(f'Unsupported type {type(x)}')


def get_module_by_name(m, name) -> nn.Module:
    return dict(m.named_modules())[name]

def frames2flow(frames):
    # t h w c, 255
    frames = np.concatenate(frames)
    # frames = frames.numpy()
    t, h, w, c = frames.shape
    import cv2
    mvs = []
    for i in range(1, len(frames)):
        next = frames[i]
        prev = frames[i - 1]
        next = np.array(next)
        prev = np.array(prev)

        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 获取运动矢量
        motion_vectors = flow.reshape(-1, 2)
        mvs.append(motion_vectors)
    mvs = [np.zeros_like(mvs[0])] + mvs
    mvs = einops.rearrange(mvs, 't (h w) two -> two t h w', h=h, w=w)
    return mvs

def repr_list(l):
    ret = repr(l)
    if len(ret) > 1000:
        ret = ret[:500] + '\n...\n' + ret[-50:]
    return ret


def color_adjustment(ref1, ref2, frames):
    # adjust frame2
    if isinstance(frames, np.ndarray):
        frames = frames.copy().astype(float)
        frames = torch.from_numpy(frames)
    else:
        frames = frames.clone().float()
    last_mean = einops.reduce(ref1.astype(float), 'h w c -> 1 1 c', reduction='mean')
    curr_mean = einops.reduce(ref2.astype(float), 'h w c -> 1 1 c', reduction='mean')
    plan = 2
    if plan == 1:
        ratio = last_mean / curr_mean
        frames *= ratio[None,:]
    elif plan == 2:
        gap = last_mean - curr_mean
        frames += gap[None,:]
    elif plan == 3:
        last_std = torch.std(ref1, dim=[-2, -3], keepdim=True)
        curr_std = torch.std(ref2, dim=[-2, -3], keepdim=True)
        frames = (frames - curr_mean[None,:]) / curr_std[None, :] * last_std[None, :] + last_mean[None, :]
    else:
        raise Exception
    return frames.clamp(min=0, max=255).type(torch.uint8).numpy()

from torch import nn
class LambdaModule(nn.Module):
    def __init__(self, lambda_expression):
        super().__init__()
        self.lambda_expression = lambda_expression
    def forward(self, *args, **kwargs):
        return self.lambda_expression(*args, **kwargs)


def get_rank():
    return int(os.environ.get('RANK', 0))

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))

def barrier():
    if dist.is_initialized():
        return dist.barrier()


def pad_arrays(arrays):
    # possible padding to guarantee the videos have the same length and size

    max_shape = tuple(max(dim) for dim in zip(*[arr.shape for arr in arrays]))

    padded_arrays = []
    for arr in arrays:
        # 计算需要填充的数量
        padding = tuple((0, max_dim - arr_dim) for max_dim, arr_dim in zip(max_shape, arr.shape))
        if isinstance(arr, np.ndarray):
            padded_arr = np.pad(arr, padding, mode='constant')
        else:
            padded_arr = torch.nn.functional.pad(arr, padding, mode='constant', value=0)
        padded_arrays.append(padded_arr)
    return padded_arrays

def collate_dicts(dics, already_has_batch_dim=False):
    ret = {}
    keys = sum((list(dic.keys()) for dic in dics), [])
    for k in keys:
        foo = []
        for dic in dics:
            assert  k in dic
            v = dic[k]
            foo.append(v)
        if isinstance(foo[0], dict):
            foo = collate_dicts(foo, already_has_batch_dim)
        elif isinstance(foo[0], torch.Tensor):
            if already_has_batch_dim:
                foo = torch.cat(foo)
            else:
                foo = torch.stack(foo)
        elif isinstance(foo[0], np.ndarray):
            if already_has_batch_dim:
                foo = np.concatenate(foo)
            else:
                foo = np.stack(foo)
        ret[k] = foo
    return ret


def fail_retry(func):
    import functools
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        while True:
            try:
                ret = func(*args, **kwargs)
                break
            except Exception as e:
                loguru.logger.error(e)
        return ret

    return wrap

import torch
from IPython.core.display import DisplayObject

class MediaArray(DisplayObject):
    verbose = False

    def __init__(self):
        super().__init__()
        self._data = None
        self.shape_pattern = None
        
    @classmethod
    def get_data_range(cls, arr):
        ret = cls._get_data_range(arr)
        if cls.verbose:
            loguru.logger.debug(f'Predict data range {ret}, (min: {arr.min()},  max: {arr.max()},  mean: {arr.mean():.2f},  std:{arr.std():.2f})')
        return ret

    @classmethod
    def _get_data_range(cls, arr):
        if (isinstance(arr, np.ndarray) and arr.dtype == np.uint8) or (isinstance(arr, torch.Tensor) and arr.dtype == torch.uint8):
            return '0-255'
        eps = 0.1
        if arr.min() >= 0-eps and arr.max() <= 1+eps:
            return '0-1'
        elif arr.min() >= -3 and arr.max() <= 3: # 因为 vae 解码出来确实有可能超出 -1, 1 的范围
            return '-1-1'
        elif arr.min() >= 0-eps and arr.max() <= 255+eps:
            return '0-255'
        # elif arr.min() >= -1-eps and arr.max() <= 1 + eps:
        raise Exception(f'Unknown data range, (min: {arr.min()},  max: {arr.max()},  mean: {arr.mean():.2f},  std:{arr.std():.2f})')

    def __or__(self, other):
        if self._data.shape == other._data.shape:
            data1 = self._data
            data2 = other._data
        else:
            # shape1 = tuple(list(self.data.shape[:-2]) + list(self.data.shape[-1:]))
            # shape2 = tuple(list(other.data.shape[:-2]) + list(other.data.shape[-1:]))
            # assert shape1 == shape2
            max_h = max(self._data.shape[-3], other._data.shape[-3])
            max_w1 = int(self._data.shape[-2] * max_h / self._data.shape[-3])
            max_w2 = int(other._data.shape[-2] * max_h / other._data.shape[-3])
            data1 = self.resize((max_h, max_w1))._data
            data2 = other.resize((max_h, max_w2))._data
        data = np.concatenate([data1, data2], axis=-2)
        return self.from_array(data)

    @property
    def shape(self):
        return self._data.shape

    @classmethod
    def to_uint8(cls, arr, data_range):
        if data_range == '0-1':
            arr = (arr * 255).clip(min=0, max=255).astype('uint8')
        elif data_range == '-1-1':
            arr = ((arr * 0.5 + 0.5) * 255).clip(min=0, max=255).astype('uint8')
        elif data_range == '0-255':
            arr = arr.clip(min=0, max=255).astype('uint8')
        return arr

    def get_array(self, lib='th', shape=None, data_range='0-255', dtype=None):
        assert shape is not None
        if lib == 'th' or lib == 'torch' or lib == 'pt':
            ret = torch.tensor(self._data)
        elif lib == 'np' or lib == 'numpy':
            ret = self._data.copy() # 不copy有时候传进cv2有问题，不知道为什么
        else:
            raise ValueError(f'Unsupported {lib}')
        if data_range == '0-1':
            ret = ret / 255.
        elif data_range == '-1-1':
            ret = (ret / 255. - 0.5) / 0.5
        else:
            assert data_range == '0-255'
        if dtype is None:
            if data_range == '0-255':
                dtype = 'uint8'
            else:
                dtype = 'float'
        if dtype is not None:
            if isinstance(ret, torch.Tensor):
                dic = {'uint8':torch.uint8, 'float':torch.float32, 'double': torch.float64}
                ret = ret.type(dic[dtype])
            elif isinstance(ret, np.ndarray):
                ret = ret.astype(dtype)
        return einops.rearrange(ret, f'{self.shape_pattern} -> {shape}')

class ImageArray(MediaArray):

    def psnr(self, other):
        try:
            from piq import psnr
            return psnr(
                self.get_array('th', 'h w c', '0-1')[None],
                other.get_array('th', 'h w c', '0-1')[None],
            ).item()
        except ImportError:
            raise ImportError("piq package is required for PSNR calculation. Install with: pip install piq --no-deps")

    @classmethod
    def guess_shape_pattern(cls, shape):
        ret = cls._guess_shape_pattern(shape)
        if cls.verbose:
            loguru.logger.debug(f'Guess shape {shape} -> {ret}')
        return ret
        
    @classmethod
    def _guess_shape_pattern(cls, shape):
        ret = ''
        if len(shape) == 4:
            ret = 'b '
            shape = shape[1:]
        if len(shape) == 3: # c t h w, t h w c
            if shape[-1] in [3, 4]:
                return ret + 'h w c'
            elif shape[0] in [3, 4]:
                return ret + 'c h w'
        raise ValueError(shape)

    @classmethod
    def from_array(cls, arr, *, shape_hint=None, data_range_hint=None): # 必须是 t h w c 或者 b c t h w
        img = ImageArray()
        if isinstance(arr, Image.Image):
            arr = np.array(arr)
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if data_range_hint is None:
            data_range = cls.get_data_range(arr)
        else:
            data_range = data_range_hint

        arr = cls.to_uint8(arr, data_range)

        if shape_hint is not None:
            shape_pattern = shape_hint
        else:
            shape_pattern = cls.guess_shape_pattern(arr.shape)
        arr = einops.rearrange(arr, f'{shape_pattern} -> h w c')

        img._data = arr
        # img.shape_pattern = shape_pattern # 因为上面已经把他强行转成 h w c 了，就不用用旧的shape_pattern了
        img.shape_pattern = 'h w c'
        assert img.shape[-1] in [3, 4], f'{img.shape} is not h w c'
        return img

    @classmethod
    def open(cls, img_path):
        return cls.from_array(np.array(Image.open(img_path).convert('RGB')))

    def save(self, img_path):
        Image.fromarray(self._data).save(img_path)

    def resize(self, resolution):
        ret = ImageArray()
        return ret.from_array(np.array(Image.fromarray(self._data).resize(resolution)))

    def resize_crop(self, resolution):
        # Fix: Use the global resize_crop function instead of undefined local one
        result_img = resize_crop(Image.fromarray(self._data), resolution)
        return self.from_array(result_img)

    def compare(self, other, T=32):
        img1 = self._data
        img2 = other._data
        assert img1.shape == img2.shape
        w = img1.shape[-2]
        frames = []
        for i in range(T // 2, -1, -1):
            curr = int(w * i / T)
            frame = np.concatenate([img1[:, :curr], np.ones_like(img1[:, :1]) * 255, img2[:, curr + 1:]], axis=1)
            frames.append(frame)
        for i in range(0, T):
            curr = int(w * i / T)
            frame = np.concatenate([img1[:, :curr], np.ones_like(img1[:, :1]) * 255, img2[:, curr + 1:]], axis=1)
            frames.append(frame)
        for i in range(T - 1, -1, -1):
            curr = int(w * i / T)
            frame = np.concatenate([img1[:, :curr], np.ones_like(img1[:, :1]) * 255, img2[:, curr + 1:]], axis=1)
            frames.append(frame)
        return VideoArray.from_array(frames)

    def to_notebook(self): # t h w c
        return self.to_pil()

    def to_pil(self):
        return Image.fromarray(self._data)

    def _repr_png_(self):
        return self.to_notebook()._repr_png_()

class VideoArray(MediaArray):# _data 永远是 t h w c

    def psnr(self, other):
        try:
            from piq import psnr
            return psnr(
                self.get_array('th', 't h w c', '0-1'),
                other.get_array('th', 't h w c', '0-1'),
            ).item()
        except ImportError:
            raise ImportError("piq package is required for PSNR calculation. Install with: pip install piq --no-deps")

    jupyter_display = 'mp4'
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self._data) # _data 永远是 t h w c
        
    @classmethod
    def guess_shape_pattern(cls, shape):
        ret = cls._guess_shape_pattern(shape)
        if cls.verbose:
            loguru.logger.debug(f'Guess shape {shape} -> {ret}')
        return ret
        
    @classmethod
    def _guess_shape_pattern(cls, shape):
        ret = ''
        if len(shape) == 5:
            ret = 'b '
            shape = shape[1:]
        if len(shape) == 4: # c t h w, t h w c
            if shape[-1] in [3, 4]:
                return ret + 't h w c'
            elif shape[0] in [3, 4]:
                return ret + 'c t h w'
        raise ValueError(shape)

    def compare(self, other):
        img1 = self._data
        img2 = other._data
        assert img1.shape[0] == img2.shape[0]
        T = img1.shape[0]
        assert img1.shape == img2.shape
        w = img1.shape[-2]
        frames = []
        left_frames = []
        right_frames = []
        for i in range(T // 2, -1, -1):
            curr = int(w * i / T)
            frame = np.concatenate([img1[i, :, :curr], np.ones_like(img1[i, :, :1]) * 255, img2[i, :, curr + 1:]], axis=1)
            frames.append(frame)
            left_frames.append(img1[i])
            right_frames.append(img2[i])
        for i in range(0, T):
            curr = int(w * i / T)
            frame = np.concatenate([img1[i, :, :curr], np.ones_like(img1[i, :, :1]) * 255, img2[i, :, curr + 1:]], axis=1)
            frames.append(frame)
            left_frames.append(img1[i])
            right_frames.append(img2[i])
        for i in range(T - 1, -1, -1):
            curr = int(w * i / T)
            frame = np.concatenate([img1[i, :, :curr], np.ones_like(img1[i, :, :1]) * 255, img2[i, :, curr + 1:]], axis=1)
            frames.append(frame)
            left_frames.append(img1[i])
            right_frames.append(img2[i])
        cmp_vid = VideoArray.from_array(frames)
        left_vid = VideoArray.from_array(np.stack(left_frames))
        right_vid = VideoArray.from_array(np.stack(right_frames))
        return left_vid | cmp_vid | right_vid

    @classmethod
    def from_array(cls, arr, *, data_range_hint=None, shape_hint=None): # 必须是 t h w c 或者 b c t h w
        v = VideoArray()
        if isinstance(arr, list):
            if len(arr) and isinstance(arr[0], np.ndarray):
                arr = np.stack(arr)
            elif len(arr) and isinstance(arr[0], torch.Tensor):
                arr = torch.stack(arr)
            elif len(arr) and isinstance(arr[0], Image.Image):
                arr = [np.array(frame) for frame in arr]
                arr = np.stack(arr)
            elif len(arr) and isinstance(arr[0], ImageArray):
                arr = [frame._data for frame in arr]
                arr = np.stack(arr)
            else:
                raise ValueError(f'Unsupported frame type {type(arr[0])}')
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if data_range_hint is None:
            data_range = cls.get_data_range(arr)
        else:
            data_range = data_range_hint
        arr = cls.to_uint8(arr, data_range)
        if len(arr.shape) == 4:
            # v.shape_pattern = 't h w c'
            if shape_hint is not None:
                shape_pattern = shape_hint
            else:
                shape_pattern = cls.guess_shape_pattern(arr.shape)
            arr = einops.rearrange(arr, f'{shape_pattern} -> t h w c')
            v.shape_pattern = 't h w c'
            # assert v._data.shape[-1] in [3, 4], f"{v.shape}"
        else:
            assert len(arr.shape) == 5, f'{arr.shape}'
            if shape_hint is not None:
                shape_pattern = shape_hint
            else:
                shape_pattern = 'b c t h w'
            v.shape_pattern = shape_pattern
            # v._data = arr
            assert v._data.shape[1] in [3, 4], f"{v._data.shape}"
        v.original_shape_pattern = shape_pattern
        v.original_data_range = data_range
        v._data = arr
        return v

    @classmethod
    def open(cls, vid_path):
        return cls.from_path(vid_path)

    @classmethod
    def from_path(cls, vid_path):
        import imageio
        return VideoArray.from_array(imageio.mimread(vid_path, memtest=False))

    def save(self, path, **kwargs):
        if len(self._data) == 1: # 如果是图片，直接存成 jpeg
            Image.fromarray(self.get_array('np', shape='t h w c', data_range='0-255')[0]).save(str(Path(path).with_suffix('.jpg')))
            return
        # default_kwags = dict( quality=9, codec='mjpeg',pixelformat='yuvj444p') # codec='mjpeg',pixelformat='yuvj444p')
        default_kwags = dict(quality=9)
        default_kwags = {'fps': 8, 'codec': 'h264'}
        default_kwags.update(kwargs)
        import imageio
        imageio.mimwrite(path, self._data, **default_kwags)

    def resize(self, resolution):
        from torchvision.transforms import _functional_video as VF
        resized_arr = VF.resize(self.get_array('th', 'c t h w', dtype='float'), (resolution, resolution) if isinstance(resolution, int) else tuple(resolution), interpolation_mode="bilinear")
        return VideoArray.from_array(resized_arr)


    def center_crop(self, resolution):
        import torchvision.transforms._transforms_video as transforms_video
        return VideoArray.from_array(transforms_video.CenterCropVideo(resolution)(self.get_array('th', 'c t h w', dtype='float')))

    def resize_crop(self, resolution):
        return self.proper_resize_centercrop(resolution)

    def proper_resize_centercrop(self, resolution):
        resize_resolution = get_proper_resize_size_by_size(self._data.shape[-3:-1], resolution)
        return self.resize(resize_resolution).center_crop(resolution)

    def to_notebook(self, suffix='mp4', **kwargs): # t h w c
        import tempfile, imageio
        from IPython.display import Video, Image
        f = tempfile.NamedTemporaryFile(suffix=f'.{suffix}')
        if 'fps' not in kwargs:
            kwargs['fps'] = 32
        if 'codec' not in kwargs:
            kwargs['codec'] = 'h264'
        imageio.mimwrite(f.name, self._data, **kwargs)
        if suffix == 'mp4':
            ret = Video(f.name, embed=True)
        else:
            ret = Image(f.name, embed=True)
        self.f = f # 避免文件被删
        return ret

    def _repr_html_(self):
        # https://notebook.community/ngoldbaum/RunNotebook/example/source/custom-display-logic
        if self.jupyter_display == 'mp4':
            return self.to_notebook(self.jupyter_display)._repr_html_()

    def _repr_mimebundle_(self, include=None, exclude=None):
        if self.jupyter_display == 'gif':
            return self.to_notebook(self.jupyter_display)._repr_mimebundle_(include, exclude)

def assert_distribution(img, data_range):
    if isinstance(img, list):
        min_val = min([frame.min() for frame in img])
        max_val = max([frame.max() for frame in img])
        mean_val = sum([frame.mean() for frame in img]) / len(img)
    else:
        min_val, max_val, mean_val = img.min(), img.max(), img.mean()
    if data_range == '0-255':
        assert min_val >= -0.1 and max_val < 256
        if mean_val < 10:
            loguru.logger.warning(f'Asserting data range within [0, 255], but the data is like mean:{mean_val}, min:{min_val}, max:{max_val}. Is it a black image?')
    elif data_range == '0-1':
        assert min_val >= -0.1 and max_val < 1.1
    elif data_range == '-1-1':
        assert min_val >= -1.1 and max_val < 1.1
    else:
        raise Exception(f'Unknown data range {data_range}')


def inflatedconv3d(x, weight, bias, padding=0, dilation=1, stride=1, groups=1):
    from torch.nn import functional as F
    from einops import rearrange

    video_length = x.shape[2]
    x = rearrange(x, "b c f h w -> (b f) c h w")
    x = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
    return x


from PIL import Image
import base64
import tempfile
import imageio


class RemoteCall:
    TYPE_IMG = 'image'
    TYPE_VIDEO = 'video'
    TYPE_TENSOR = 'tensor'
    TYPE_BUILTIN = 'builtin'

    def __init__(self, url=None):
        self.url = url

    @classmethod
    def wrap_obj(self, obj, with_type=False):
        if isinstance(obj, Image.Image):
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg')
            obj.convert('RGB').save(temp_file.name)
            b64 = base64.b64encode(Path(temp_file.name).read_bytes()).decode('utf-8')
            if with_type:
                return (self.TYPE_IMG, b64)
            else:
                return b64
        elif isinstance(obj, torch.Tensor):
            temp_file = tempfile.NamedTemporaryFile(suffix='.pt')
            torch.save(obj, temp_file.name)
            b64 = base64.b64encode(Path(temp_file.name).read_bytes()).decode('utf-8')
            if with_type:
                return (self.TYPE_TENSOR, b64)
            else:
                return b64
        elif (
                isinstance(obj, list) and len(obj) > 0 and
                (isinstance(obj[0], torch.Tensor) or isinstance(obj[0], np.ndarray))
        ):
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4')
            VideoArray.from_array(obj).save(temp_file.name)
            b64 = base64.b64encode(Path(temp_file.name).read_bytes()).decode('utf-8')
            if with_type:
                return (self.TYPE_VIDEO, b64)
            else:
                return b64
        elif isinstance(obj, VideoArray):
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4')
            obj.save(temp_file.name)
            b64 = base64.b64encode(Path(temp_file.name).read_bytes()).decode('utf-8')
            if with_type:
                return (self.TYPE_VIDEO, b64)
            else:
                return b64
        elif isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, dict) or isinstance(obj, list):
            if with_type:
                return (self.TYPE_BUILTIN, obj)
            return obj
        else:
            raise ValueError(repr(obj))
    @classmethod
    def unwrap_obj(self, obj):
        data_type, obj = obj
        return self.unwrap_obj_given_type(obj, data_type)

    @classmethod
    def unwrap_obj_given_type(self, obj, data_type):
        if data_type == self.TYPE_BUILTIN:
            return obj
        elif data_type == self.TYPE_IMG:
            return Image.open(io.BytesIO(base64.b64decode(obj)))
        elif data_type == self.TYPE_TENSOR:
            return torch.load(io.BytesIO(base64.b64decode(obj)), map_location='cpu')
        elif data_type == self.TYPE_VIDEO:
            vid_file = tempfile.NamedTemporaryFile(suffix='.mp4')
            vid_file.write(base64.b64decode(obj))
            vid_file.flush()
            return imageio.mimread(vid_file.name, memtest=False)
        else:
            raise ValueError(f'Unrecognized object: {repr(obj)}')


    def __call__(self, **kwargs):
        dic = {k:self.wrap_obj(v, with_type=True) for k, v in kwargs.items()}
        import requests
        headers = { "Content-type": "application/json" ,
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
                    # 'Connection': 'close',
                    # 'Accept-Encoding': 'identity',
                    }
        # logging.basicConfig(level=logging.DEBUG) # 之前httplib就能访问成功，用requests就一直 access denied, 靠这一行才发现是因为proxy的问题。。。
        proxies = {
            "http": "",
            "https": "",
        }
        resp = requests.post(self.url, json=dic, headers=headers, verify=True, allow_redirects=False, proxies=proxies)
        # resp = myrequests.post(self.url, json=dic, headers=headers)
        return resp.json()

def render_config_file(path, **kwargs):
    import jinja2
    import tempfile
    with open(path, 'r', encoding='utf-8') as input_f:
        render_conf_str = jinja2.Template(source=input_f.read()).render(**kwargs)
    ret = tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='yaml')
    ret.write(render_conf_str)
    ret.flush()
    return ret

class myrequests:
    @staticmethod
    def post(url, json, headers):
        import http.client as httplib
        import json as json_lib
        import re
        reg = r'((\d|\.)+):(\d+)(/.*)'
        ip, _, port, req_url = re.search(reg, url).groups()
        print(ip, port)
        print(url)
        port = int(port)
        conn = httplib.HTTPConnection(ip, port)
        conn.request('POST', req_url, json_lib.dumps(json), headers)
        return conn.getresponse()


class BucketPreprocessor:

    def __init__(self, height=384, width=640):
        original_height, original_width = height, width
        max_size = max(width, height)
        standard_pixel_num = height * width
        max_ar = 2.5
        self.buckets = []
        base = 64
        if global_config.pixel_space_training:
            base = 8
        trial_width = 256
        if global_config.pixel_space_training:
            base = 32
        while trial_width < max_size:
            trial_height = (standard_pixel_num / trial_width) // base * base
            trial_height = int(trial_height)
            if trial_height < trial_width:
                break
            if trial_height / trial_width > max_ar:
                trial_height = int(max_ar * trial_width // base * base)
            self.buckets.append((trial_height, trial_width))
            trial_width += base
        for buc in self.buckets[:]:
            self.buckets.append((buc[1], buc[0]))

        import math
        sqaure_h = int(math.sqrt(standard_pixel_num) // base * base)
        self.buckets.append((sqaure_h, sqaure_h))
        self.buckets = list(set(self.buckets))

        # 增大原始尺寸训练的概率
        self.buckets = [(original_height, original_width)] * 6 + self.buckets

    def get_bucket(self, rank):
        bucket_idx = rank % len(self.buckets)
        return self.buckets[bucket_idx]


    def __call__(self, pixel_values, global_rank):
        # pixel_values: t c h w (-1, 1)
        bucket_idx = global_rank % len(self.buckets)
        size = self.buckets[bucket_idx]
        return resize_crop(pixel_values, size)


def resize_crop(img, size):
    # from PIL import Image
    # if isinstance(img, np.ndarray):
    #     img = Image.fromarray(img)
    # W, H = img.size
    # h, w = size
    # r = max(h / H, w / W)
    # new_size = int(H * r), int(W * r)

    new_size = get_proper_resize_size(img, size)
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(new_size),
        T.CenterCrop(size),
    ])(img)

class SmartResize:
    def __init__(self, resolution):
        self.resolution = resolution
    def __call__(self, img):
        res = get_proper_resize_size(img, self.resolution)
        from torchvision import transforms as T
        return T.Resize(res)(img)

def get_proper_resize_size_by_size(curr_size, size):
    H, W = curr_size
    h, w = size
    r = max(h / H, w / W)
    return int(H * r), int(W * r)

def get_proper_resize_size(img, size):
    from PIL import Image
    if isinstance(size, int):
        size = (size, size)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if isinstance(img, Image.Image):
        W, H = img.size
    elif isinstance(img, np.ndarray):
        if img.shape[-1] in [3, 4]:
            H, W = img.shape[-3:-1]
        else:
            H, W = img.shape[-2:]
    elif isinstance(img, torch.Tensor):
        H, W = img.shape[-2:]
    else:
        raise Exception
    h, w = size
    r = max(h / H, w / W)
    return int(H * r), int(W * r)

if __name__ == '__main__':
    dics = [{'a': torch.randn(5, 10)}, {'a':torch.zeros(5, 10)}]
    r = collate_dicts(dics)


# def __position():
#     frame = inspect.currentframe().f_back.f_back
#     filename = frame.f_code.co_filename
#     lineno = frame.f_lineno
#     ret = f'{filename}:{lineno}'
#     # print(ret)
#     return ret

def __position(depth=0):
    import sys
    frame = sys._getframe(depth + 1)
    file = frame.f_code.co_filename
    line = frame.f_lineno
    return f'{file}:{line}'


def get_arguments(func, local_vars, strict=False):
    import inspect
    ret = {}
    for param in inspect.signature(func).parameters.values():
        if param.name in local_vars:
            ret[param.name] = local_vars[param.name]
        else:
            if strict:
                raise Exception(f'Missing {param.name} in local variable, (function: {func})')
    return ret

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def bing_cache(max_count):
    def decorator(func):
        from collections import Counter
        dic = Counter()
        from functools import wraps

        @wraps(func)
        def wrapper(*args):
            if dic[args] % max_count != 0:
                dic[args] += 1
                return
            if dic[args] > 0:
                ret = func(args[0], args[1] + f'  x {max_count}', *args[2:])
            else:
                ret = func(*args)
            dic[args] += 1
            return ret
        return wrapper
    return decorator


# @lru_cache(None)
@bing_cache(10)
def __log_less(pos, msg, level, additional_depth):
    level = level.lower()
    # `depth` Specify which stacktrace should be used to contextualize the logged message.
    # This is useful while using the logger from inside a wrapped function to retrieve worthwhile information.
    # See: loguru/_logger.py:1298
    getattr(loguru.logger.opt(depth=3 + additional_depth), level)(msg)


@bing_cache(99999)
def __log_once(pos, msg, level, additional_depth):
    level = level.lower()
    # `depth` Specify which stacktrace should be used to contextualize the logged message.
    # This is useful while using the logger from inside a wrapped function to retrieve worthwhile information.
    # See: loguru/_logger.py:1298
    getattr(loguru.logger.opt(depth=3 + additional_depth), level)(msg)

# Keep track of 10 different messages and then warn again
def log_less(msg, level='debug', additional_depth=0):
    # 用 1 找到调用 log_once 的地方
    # 加上 additional 表示  `调用 log_once 的地方` 可能被真正需要用的地方调用
    __log_less(__position(1 + additional_depth), msg, level, additional_depth)

def log_once(msg, level='critical', additional_depth=0):
    # 用 1 找到调用 log_once 的地方
    # 加上 additional 表示  `调用 log_once 的地方` 可能被真正需要用的地方调用
    __log_once(__position(1 + additional_depth), '[log once]: ' + msg, level, additional_depth)

if __name__ == '__main__':
    for i in range(10):
        log_less('xxxx')


def eval(s, type):
    if type == str:
        return s
    if type in [float, int, list]:
        ret = builtins.eval(s)
        if isinstance(ret, type):
            return ret
        raise ValueError(f'Invalid argument {s} for type {type}')
    if type == bool:
        if s.lower() in ['true', 'false']:
            s = s[0].upper() + s[1:]
            return builtins.eval(s)
        return bool(int(s)) # int
    raise ValueError(f'Invalid type {type}')


def ddp_set_trace():
    if get_local_rank() == 0:
        import pdb
        pdb.set_trace()


# @contextmanager
# def temporary_modification(msg):
#     temporary_modification_log(f'[临时修改，记得恢复回去]：{msg}')
#     yield

def prompt_to_file_name(prompt):
    return prompt.replace(':', '.').replace('/', '_')[:50]

def temporary_modification_log(msg, enable=True):
    # enable = False
    if enable:
        # loguru.logger.critical(f'[临时修改，记得恢复回去]：{msg}')
        log_less(f'[临时修改，记得恢复回去]：{msg}', 'critical', additional_depth=1)
    return enable

if __name__ == '__main__':
    assert eval('true', bool) == True
    assert eval('false', bool) == False
    assert eval('1', bool) == True
    assert eval('2', bool) == True
    assert eval('0', bool) == False
    assert eval('1', int) == 1
    assert eval('1.5', float) == 1.5

def is_controlnet(m):
    return hasattr(m, 'is_controlnet') and m.is_controlnet

def format_keys(keys, depth=5, prefix=None):
    from collections import defaultdict
    dic = defaultdict(list)
    def find_nth(haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start+len(needle))
            n -= 1
        return start
    for k in keys:
        pos = find_nth(k, '.', depth)
        dic[k[:pos]].append(k)
    ret = []
    n = 5
    for k, v in dic.items():
        if len(v) < n:
            ret += v
        else:
            ret += v[:n]
            ret += [k + f': [{len(v) - n} more params]']
    if prefix:
        ret = [f'{prefix}: {s}' for s in ret]
    return '\n'.join(ret)

def is_close(a, b, atol=1e-8):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    a, b = a.float(), b.float()
    return torch.allclose(a, b, atol=atol)


def load_sd_file(path:str):
    path = possible_path_map(path)

    if path == '':
        loguru.logger.warning('尝试读取空的文件，或许是不想加载任何checkpoint, 直接返回一个空dict')
        return {}
    def _ends_with(s, suffix_list):
        if isinstance(suffix_list, str):
            suffix_list = [suffix_list]
        for sf in suffix_list:
            if s.endswith(sf):
                return True
        return False
    if _ends_with(path, ['.pt', '.ckpt', '.bin']):
        return torch.load(path, map_location='cpu')
    elif path.endswith('.safetensors'):
        import safetensors
        d = safetensors.safe_open(path, 'torch')
        loguru.logger.info('Use safetensor')
        return {k: d.get_tensor(k) for k in d.keys()}
    else:
        from pathlib import Path
        raise Exception(f'Unsupported sd suffix {Path(path).suffix}')

def remove_possible_module_from_sd(sd):
    if 'state_dict' in sd:
        sd = sd['state_dict']
    return {k.replace('module.', ''):v for k, v in sd.items()}

def load_state_dict(model, state_dict, strict=True, ignore_shape_mismatch=False):
    import loguru
    additional_unexpected = []
    if ignore_shape_mismatch:
        assert not strict
        model_state_dict = model.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    loguru.logger.warning(
                        f"Skip loading parameter: {k}, "
                        f"" f"required shape: {model_state_dict[k].shape}, "
                        f"" f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    additional_unexpected.append(k)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    unexpected.extend(additional_unexpected)
    return missing, unexpected


def load_state_dict_from_path(model, sd_path):
    loguru.logger.info(f'Loading from {sd_path}')
    sd = load_sd_file(sd_path)
    sd = remove_possible_module_from_sd(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) == 0 and len(unexpected) == 0:
        loguru.logger.info(f'Perfect match. Parameters loaded')
    else:
        from hy_parallelism.utils import get_missing_unexpected_str
        loguru.logger.warning(get_missing_unexpected_str(missing, unexpected))
        # if len(missing) != 0:
        #     loguru.logger.warning(f'Missing {format_keys(missing)}')
        # else:
        #     loguru.logger.info('No Missing')
        # if len(unexpected) != 0:
        #     loguru.logger.warning(f'Unexpected {format_keys(unexpected)}')
        # else:
        #     loguru.logger.info('No Unexpected')


def get_checkpoint_path(save_dir, step=None):
    files = list(Path(save_dir).glob('*.ckpt'))
    files = sorted(files, key=lambda x: os.path.getmtime(x))
    if len(files) >= 3:
        to_be_delete = files[0]
    else:
        to_be_delete = None
    # last = files[-1].name
    if step is None:
        # idx = 0
        # import re
        # re.search('checkpoint_(\d+).ckpt', last).groups()[0]
        import datetime
        now = datetime.datetime.now()
        return to_be_delete, f'{save_dir}/checkpoint_{now:%Y-%m-%d %H-%M-%S}.ckpt'
    else:
        return to_be_delete, f'{save_dir}/checkpoint_{step}.ckpt'


def safe_save(obj, path):
    if Path(path).exists():
        new_file = path+'.safe'
        bak_file = path + '.bak'
        torch.save(obj, new_file)

        # torch.load(new_file, map_location='cpu')

        shutil.move(path, bak_file)
        shutil.move(new_file, path)
        Path(bak_file).unlink()
    else:
        torch.save(obj, path)

def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):

    train_len = 16
    if max_len > train_len:
        ratio = max_len / train_len
        ratio = 3
    else:
        ratio = 1
    log_less(f'Use NTK alpha {ratio}')
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    base = 10000

    # NTK
    # https://kexue.fm/archives/9706
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=b80b3f37
    # base = base * ratio ** (output_dim / (output_dim - 2))  # Base change formula



    theta = torch.pow(base, -2 * ids / output_dim)

    # (max_len, output_dim//2)
    embeddings = position * theta # 即公式里的：pos / (10000^(2i/d))

    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

    # (bs, head, max_len, output_dim)
    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings

class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
    # https://kexue.fm/archives/8265
    # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
    # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
    # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
    rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
        query_layer
    )
    query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
    # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
    rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
    key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
    if value_layer is not None:
        # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
        rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
            value_layer
        )
        value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
        return query_layer, key_layer, value_layer
    return query_layer, key_layer

def RoPE(q, k, pos_emb=None):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    if pos_emb is None:
        pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)


    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了



    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos

    return q, k


def get_valid_frame_strides(full_length, n_frame):
    # fulllen - 1 >= stride * (n_frame - 1)
    # =>  stride >= (fulllen - 1) / (n_frame - 1)
    if n_frame == 1:
        return set(range(10)) # 多少都行，避免后面报错
    max_stride = (full_length - 1) // (n_frame - 1)
    return set(range(1, max_stride + 1))

def sample_frame(full_len, n_frame, stride):
    import random
    min_len = stride * (n_frame - 1) + 1
    pos = random.randint(0, full_len - min_len)
    ret = list(range(pos, pos + stride * (n_frame - 1) + 1, stride))
    assert len(ret) == n_frame, f'{pos}, {pos + stride * (n_frame - 1) + 1}'
    return ret


def sample_frame_idx_from_vr(vr, sample_len, default_stride, return_frame_stride=True):
    import random
    possible_strides = get_valid_frame_strides(len(vr), sample_len)

    if global_config.dynamic_fps:
        # overlap = possible_strides & set(range(1, 25))
        overlap = possible_strides & set(range(4, 64+1, 4))
        frame_stride = random.choice(list(overlap))
    else:
        frame_stride = default_stride
        if sample_len != 1 and frame_stride not in possible_strides:
            raise Exception(f'当前视频 (长度 {len(vr)}） 不能按照 {default_stride} 的间隔抽帧')
    sample_index = sample_frame(len(vr), sample_len, frame_stride)

    assert(len(sample_index) == sample_len), f'current clip_length={len(sample_index)}, target clip_length={sample_len}, {sample_index}'

    if return_frame_stride:
        return sample_index, frame_stride
    else:
        return sample_index


def Fourier_filter(x, threshold, scale):
    from torch import fft
    x = x.float()
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, T, H, W = x_freq.shape
    mask = torch.ones((B, C, T, H, W)).cuda()

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered

def memory_efficient_attention(query, key, value, attn_bias, num_heads):
    from memory_efficient_attention_pytorch.memory_efficient_attention import memory_efficient_attention
    import importlib
    meamodule = importlib.import_module('memory_efficient_attention_pytorch.memory_efficient_attention')
    meamodule.checkpointed_summarize_qkv_chunk = meamodule.summarize_qkv_chunk
    func = lambda x: einops.rearrange(x, '(b h) n c -> b h n c', h=num_heads)
    query = func(query)
    key = func(key)
    value = func(value)
    hidden_states = memory_efficient_attention(query, key, value, attn_bias=attn_bias)
    hidden_states = einops.rearrange(hidden_states, 'b h n d -> b n (h d)')
    return hidden_states


def _get_path_map(path):
    if 'apdcephfs_sh' in path: # 已经是上海了，别映射了
        return path
    base = '/mnt/apdcephfs_sh1/share_301124792/kevinkhwu/'
    new_path = os.path.join(base, str(Path(path).absolute())[1:])
    return new_path
def possible_path_map(path):
    if hasattr(global_config, 'is_cq') and global_config.is_cq:
        return path # uncomment this line to disable mapping for cq server
    new_path = _get_path_map(path)
    if Path(new_path).exists():
        loguru.logger.info(f'上海映射： {path} -> {new_path}')
        return new_path
    else:
        if '_sh1' not in path:
            loguru.logger.warning(f'无法映射到上海ceph {path}')
    return path


def to_sh(path):
    loguru.logger.info(f'Copying {path}')
    new_path = _get_path_map(path)
    Path(new_path).parent.mkdir(exist_ok=True, parents=True)
    if Path(new_path).exists():
        loguru.logger.warning(f'{new_path} exists')
        return
    if Path(path).is_dir():
        shutil.copytree(path, new_path)
    else:
        shutil.copy(path, new_path)
    loguru.logger.info(f'copy {path} to {new_path}')

if __name__ == '__main__':
    ...
    files = [
        '/mnt/apdcephfs/share_1367250/jacobkong/gitpackages/huggingface_models/IP-Adapter/',
        '/mnt/apdcephfs_cq2/share_1367250/kevinkhwu/pretrained_models/ip-adapter-plus_sd15.bin',
        '/mnt/apdcephfs_cq2/share_1367250/kevinkhwu/pretrained_models',
    ]
    for f in files:
        to_sh(f)

import socket
import fcntl
import struct

def get_ip_address(ifname):
    ifname = ifname.encode('utf-8')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])

def get_local_ip():
    try:
        return get_ip_address('bond1')  # '192.168.0.110'
    except:
        return get_ip_address('eth1')

def enforce_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
            alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas

def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def getattr_bing(obj, name:str):
    try:
        return obj[name]
    except Exception as e:
        ...
    try:
        import builtins
        return builtins.getattr(obj, name)
    except Exception as e:
        ...
    raise AttributeError(f'No attribute named {name} for {type(obj)}')
def getattr_by_str(name):
    objs = name.split('.')
    frame = inspect.currentframe().f_back
    locals_dict = frame.f_locals
    o = locals_dict.get(objs[0])
    # o = builtins.eval(objs[0])
    for i in range(1, len(objs)):
        o = getattr_bing(o, objs[i])
    return o

def get_obj_from_str(string, reload=False, invalidate_cache=True):
    import importlib
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_keep_ar_resize_size(old_size, curr_size):
    ar = old_size[0] / old_size[1]
    # x / y == ar
    # x >= curr_size[0]
    # y >= curr_size[1]
    # y * ar >= curr_size[0]
    # y >= curr_size[1]
    y = max(curr_size[0] / ar, curr_size[1])
    x = ar * y
    x, y = map(int, (x, y))
    return (x, y)
def scale_bounding_box(bounding_box, original_image_size, target_image_size):
    """
    计算缩放后的边界框坐标。
    """
    x1, y1, x2, y2 = bounding_box
    original_width, original_height = original_image_size
    target_width, target_height = target_image_size

    scale_width = target_width / original_width
    scale_height = target_height / original_height

    new_x1 = x1 * scale_height
    new_y1 = y1 * scale_width
    new_x2 = x2 * scale_height
    new_y2 = y2 * scale_width

    return new_x1, new_y1, new_x2, new_y2


def dump_json(dic, path, overwrite=False):
    import json
    if not overwrite and Path(path).exists():
        raise Exception(f'{path} 存在')
    with open(path, 'w') as f:
        json.dump(dic, f)

def load_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)


class timeout_resp:
    def __init__(self, resp, timeout=5):
        self.resp = resp
        self.timeout = timeout

    def __enter__(self):
        self.resp.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.resp.__exit__(exc_type, exc_val, exc_tb)
        return self
    def read(self):
        start_time = time.time()
        buffer = io.BytesIO()
        while True:
            chunk = self.resp.read(4096)
            if not chunk:
                break
            buffer.write(chunk)
            download_time = time.time() - start_time
            if download_time > self.timeout:
                raise TimeoutError
        return buffer.getvalue()

# import timeout_decorator
class timeout_request:
    @staticmethod
    def urlopen(req, timeout):
        import urllib
        return timeout_resp(urllib.request.urlopen(req, timeout=timeout))

def get_synced_object(func, exec_rank=0):
    if get_rank() == exec_rank:
        obj_list = [func()]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0, device=torch.device(get_local_rank()))
    return obj_list[0]


class StreamWrapper:
    def __init__(self, fpath=None, std_what='stdout', shared_file_descriptor=None):
        assert std_what in ['stdout', 'stderr']
        self.console = getattr(sys, std_what)
        self.file = open(fpath, 'w')
        self.shared_file_descriptor = shared_file_descriptor
        print(fpath)

    def __getattr__(self, item):
        return getattr(self.console, item)

    # def __del__(self):
    #     self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write_to_file(self, f, msg):
        if f is not None:
            f.write(msg)
            # f.flush()

    def write(self, msg):
        self.console.write(msg)
        self.write_to_file(self.file, msg)
        self.write_to_file(self.shared_file_descriptor, msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
        if self.shared_file_descriptor is not None:
            self.shared_file_descriptor.flush()
            # try:
            #     import os
            #     os.fsync(self.file.fileno())
            # except:
            #     ...

    def close(self):
        self.flush()
        if self.file is not None:
            self.file.close()
        return self.console

class UnifiedOutputWrapper:

    def __init__(self, stdout_path, stderr_path, shared_path):
        self.check_path(stdout_path)
        self.check_path(stderr_path)
        self.check_path(shared_path)
        self.stdout_path, self.stderr_path, self.shared_path = stdout_path, stderr_path, shared_path
        self.shared_file = open(shared_path, 'w')

    def check_path(self, path):
        parent = Path(path).parent
        if parent.exists() and not parent.is_dir():
            raise Exception(f'{parent.as_posix()} is not a directory')
        if not parent.exists():
            print(f'Create directory {parent.as_posix()}...')
            parent.mkdir(exist_ok=True, parents=True)

    def configure(self):
        sys.stdout = StreamWrapper(self.stdout_path, 'stdout', shared_file_descriptor=self.shared_file)
        sys.stderr = StreamWrapper(self.stderr_path, 'stderr', shared_file_descriptor=self.shared_file)

    def reset(self):
        stdout_console = sys.stdout.close()
        stderr_console = sys.stderr.close()
        self.shared_file.close()
        sys.stdout = stdout_console
        sys.stderr = stderr_console



from typing import Callable


def replace_module(model, is_target_module:Callable, get_alternative:Callable):
    def __replace_module(model, full_name):
        for name, child in model.named_children():
            if is_target_module(full_name, child):
                new_module = get_alternative(full_name, child)
                setattr(model, name, new_module)
            __replace_module(child, full_name + '.' + name)
    __replace_module(model, '')

def resize(img, factor):
    from torch.nn import functional as F
    if len(img.shape) > 4:
        t = img.shape[-3]
        img = einops.rearrange(img, '... c t h w -> (... t) c h w')
        img = F.interpolate(img, scale_factor=factor, mode='bilinear', align_corners=False)
        return einops.rearrange(img, '(b t) c h w -> b c t h w', t=t)
    else:
        img = F.interpolate(img, scale_factor=factor, mode='bilinear', align_corners=False)
        return img


def rebuild_dtensor(
        dtensor,
        real_device_mesh,
        real_src_placement, real_target_placement=None,
):

    from torch.distributed.tensor import DTensor
    if real_device_mesh != dtensor.device_mesh or real_src_placement != dtensor.placements:
        dtensor = DTensor.from_local(dtensor.to_local(), device_mesh=real_device_mesh, placements=real_src_placement)
    if real_target_placement is not None and real_target_placement != real_src_placement:
        dtensor = dtensor.redistribute(real_device_mesh, real_target_placement)
    return dtensor


# 只存训练配置。模型配置尽量写在配置文件里
# temp 开头的都是为了快速实验临时加进来的，后面验证有效后，记得重构代码  把其写进配置文件里
class __GlobalConfigClass():


    def update_by_dict(self, dic):
        dic:dict
        for k, v in dic.items():
            setattr(self, k, v)


    def to_dict(self):
        dic = {}
        max_len = 0
        for k in dir(self):
            if not k.startswith('__'):
                v = getattr(self, k)
                from typing import Callable
                if not isinstance(v, Callable):
                    dic[k] = v
        return dic

    def dump(self, path):
        dic = self.to_dict()

        import json
        def remove_unserializable_pairs(data):
            serializable_data = {}
            for key, value in data.items():
                try:
                    json.dumps({key:value})
                    serializable_data[key] = value
                except:
                    print(f"无法序列化的键值对: {key} - {value}")
            return serializable_data
        dic = remove_unserializable_pairs(dic)
        with open(path, 'w') as f:
            json.dump(dic, f, indent=4)

    @classmethod
    def from_json(cls, path):
        import json
        with open(path, 'r') as f:
            dic = json.load(f)
        obj = cls()
        obj.update_by_dict(dic)
        return obj


    def to_str(self):
        ret = ''
        dic = {}
        max_len = 0
        for k in dir(self):
            if not k.startswith('__'):
                max_len = max(max_len, len(k))
                v = getattr(self, k)
                from typing import Callable
                if not isinstance(v, Callable):
                    dic[k] = v
        for k, v in dic.items():
            ret += f'{k:<{max_len}} = {repr(v)}\n'
        return ret

    def __str__(self):
        return self.to_str()

global_config = __GlobalConfigClass()

def keep_channel_gradient_ratio(tensor, ratio, first_last):
    n = int(tensor.shape[1] * ratio)
    assert n >= 1 and tensor.shape[1] > 1
    if first_last == 'first':
        return torch.cat([tensor[:, :n], tensor[:, n:].detach()], dim=1)
    else:
        return torch.cat([tensor[:, :-n].detach(), tensor[:, -n:]], dim=1)

# if __name__ == '__main__':
#     from utils import bing_utils
#     bing_utils.copy_by_suffix('.', 'debug_copy')


import io
from typing import List
import numpy as np

# bobby 实现的支持不同backend的video reader
class VideoReader:
    def __init__(self, source, backend="pyav"):
        self.backend = backend
        self.source = source

        print(isinstance(source, bytes))
        if backend == "decord":
            import decord
            if isinstance(source, bytes):
                self.video_reader = decord.VideoReader(io.BytesIO(source))
            else:
                self.video_reader = decord.VideoReader(source)
            self.total_frames = len(self.video_reader)
        elif backend == "imageio":
            import imageio
            if isinstance(source, bytes):
                self.video_reader = imageio.get_reader(io.BytesIO(source), 'pyav')
            else:
                self.video_reader = imageio.get_reader(source, 'pyav')
            self.total_frames = self.video_reader.count_frames()
        elif backend == "pyav":
            import av
            if isinstance(source, bytes):
                self.container = av.open(io.BytesIO(source))
            else:
                self.container = av.open(source)
            self.video_stream = self.container.streams.video[0]
            self.total_frames = self.video_stream.frames
            self.frames = []
            self._load_frames()
        else:
            raise ValueError("Unsupported backend. Use 'decord' or 'pyav'.")

    def __len__(self):
        return len(self.frames)
        return self.total_frames

    # def __getitem__(self, item):
    #     return self.frames[item]

    def h(self): # t h  w c
        return self.frames[0].shape[-3]

    def w(self):
        return self.frames[0].shape[-2]


    def get_batch(self, indices: List[int]):
        if self.backend == "decord":
            return self.video_reader.get_batch(indices).asnumpy()
        elif self.backend == "pyav":
            return np.array([self.frames[i] for i in indices])
        elif self.backend == "imageio":
            return np.array(self._get_imageio_batch(indices))

    def _load_frames(self):
        for frame in self.container.decode(video=0):
            self.frames.append(frame.to_ndarray(format='rgb24'))

    def _get_imageio_batch(self, indices: List[int]):
        frames = []
        for index in sorted(indices):
            frame = self.video_reader.get_data(index)
            frames.append(frame)
        return frames


def check_random_states_diff(group):
    import random
    import numpy as np
    import torch
    import math

    random_val = gather_obj(random.random(), group=group)

    has_neq = False
    cnt = 0
    while not has_neq:
        for v in random_val[1:]:
            if not (math.isclose(v, random_val[0], rel_tol=1e-6)):
                has_neq = True
                break
        cnt += 1
        if cnt > 10:
            raise RuntimeError('随机数突然相等了？')


    has_neq = False
    cnt = 0
    while not has_neq:
        random_val = gather_obj(np.random.rand(), group=group)
        for v in random_val[1:]:
            if not (np.isclose(v, random_val[0])):
                has_neq = True
                break
        cnt += 1
        if cnt > 10:
            raise RuntimeError('随机数突然相等了？')


    has_neq = False
    cnt = 0
    while not has_neq:
        random_val = gather_obj(torch.rand(1), group=group)
        for v in random_val[1:]:
            if not (torch.allclose(v, random_val[0])):
                has_neq = True
                break
        cnt += 1
        if cnt > 10:
            raise RuntimeError('随机数突然相等了？')

def check_random_states(group, same=True):
    import random
    import numpy as np
    import torch
    import math
    if same:
        op = lambda x: x
    else:
        op = lambda x: not(x)

    random_states = gather_obj(random.getstate(), group=group)
    for s in random_states:
        for p1, p2 in zip(s, random_states[0]):
            assert op(p1 == p2)
    random_val = gather_obj(random.random(), group=group)
    for v in random_val:
        assert op(math.isclose(v, random_val[0], rel_tol=1e-6))

    random_states = gather_obj(np.random.get_state(), group=group)
    anchor = random_states[0]
    for s in random_states:
        assert op(s[0] == anchor[0])
        assert op(np.allclose(s[1], anchor[1]))
        assert op(s[2] == anchor[2])
        assert op(s[3] == anchor[3])
        assert op(s[4] == anchor[4])
    random_val = gather_obj(np.random.rand(), group=group)
    for v in random_val:
        assert op(np.isclose(v, random_val[0]))

    random_states = gather_obj(torch.random.get_rng_state(), group=group)
    anchor = random_states[0]
    for s in random_states:
        assert op(torch.equal(s, anchor))
    random_val = gather_obj(torch.randn(10, 20), group=group)
    for v in random_val:
        assert op(torch.allclose(v, random_val[0]))
    random_val = gather_obj(torch.randn((10, 20), device='cuda'), group=group)
    for v in random_val:
        assert op(torch.allclose(v.cpu(), random_val[0].cpu()))


def check_parallel_random_stats(parallel_dims):
    if parallel_dims.pp_enabled:
        check_random_states(parallel_dims.pp_mesh.get_group())
    if parallel_dims.sp_enabled:
        check_random_states(parallel_dims.sp_mesh.get_group())
    if parallel_dims.tp_enabled:
        check_random_states(parallel_dims.tp_mesh.get_group())
    if parallel_dims.dp_enabled:
        check_random_states_diff(parallel_dims.dp_mesh.get_group())

class Counter:
    ...

def increase_counter(s):
    if hasattr(Counter, s):
        setattr(Counter, s, getattr(Counter, s) + 1)
    else:
        setattr(Counter, s, 1)


_profiles = []

def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    from datetime import datetime
    TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"prof/rank{dist.get_rank()}_{timestamp}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")

def get_profiler():
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    )
    return prof

def start_one_profile():
    prof = get_profiler()
    _profiles.append(prof)
    prof.__enter__()

def stop_profiles() :
    for prof in _profiles:
        prof.__exit__(None, None, None)

def profile_step():
    for prof in _profiles:
        prof.step()

from contextlib import contextmanager

@contextmanager
def super_meta():
    old_to = nn.Module.to
    def no_op(self, device=None, dtype=None, *args, **kwargs):
        return old_to(self, dtype=dtype)
    nn.Module.to = no_op
    with torch.device('meta'):
        yield
    nn.Module.to = old_to

def assert_tensor_close(a, b, tag='', verbose=True, raise_exception=True, **kwargs):
    raise DeprecationWarning("assert_tensor_close is deprecated. Use assert_close instead.")
    try:
        result = assert_close(a, b, **kwargs)
        equal = result.get('equal', False)
        if verbose:
            if tag:
                loguru.logger.info(f"`{tag}` precision check passed")
            else:
                loguru.logger.info("Precision check passed")
        return equal
    except Exception as e:
        if not raise_exception:
            if tag:
                loguru.logger.error(f"`{tag}` precision check failed: {str(e)}")
            else:
                loguru.logger.error(f"Precision check failed: {str(e)}")
            return False
        raise e

    
def extract_mismatch_and_diff(text):
    if not text:
        return {
            'equal': True,
            'mismatch_percent': 0,
            'mismatch_count': 0,
            'compared_total': 0,
            'greatest_abs_diff': 0,
            'abs_diff_index': None,
            'atol': None,
            'greatest_rel_diff': None,
            'rel_diff_index': None,
            'rtol': None,
        }
    import re
    # Extract mismatched numbers and percentage from 'Mismatched elements:'
    percent_pat = r'Mismatched elements: (\d+) / (\d+) \(([\d\.]+)%\)'
    percent_match = re.search(percent_pat, text)
    mismatch_count = int(percent_match.group(1)) if percent_match else None
    mismatch_total = int(percent_match.group(2)) if percent_match else None
    percent = float(percent_match.group(3)) if percent_match else None

    # Extract "greatest absolute difference" value, its index, and "up to ... allowed" (atol)
    abs_diff_pat = r'Greatest absolute difference: ([\d\.eE+-]+) at index (\([^)]+\))(?: \(up to ([\d\.eE+-]+) allowed\))?'
    abs_diff_match = re.search(abs_diff_pat, text)
    abs_diff, abs_diff_index, atol = None, None, None
    if abs_diff_match:
        abs_diff = float(abs_diff_match.group(1))
        abs_diff_index = abs_diff_match.group(2)
        if abs_diff_match.group(3) is not None:
            try:
                atol = float(abs_diff_match.group(3))
            except Exception:
                atol = abs_diff_match.group(3)

    # Extract "greatest relative difference" value, its index, and "up to ... allowed" (rtol)
    rel_diff_pat = r'Greatest relative difference: ([\d\.eE+-]+) at index (\([^)]+\))(?: \(up to ([\d\.eE+-]+) allowed\))?'
    rel_diff_match = re.search(rel_diff_pat, text)
    rel_diff, rel_diff_index, rtol = None, None, None
    if rel_diff_match:
        rel_diff = float(rel_diff_match.group(1))
        rel_diff_index = rel_diff_match.group(2)
        if rel_diff_match.group(3) is not None:
            try:
                rtol = float(rel_diff_match.group(3))
            except Exception:
                rtol = rel_diff_match.group(3)

    return {
        'equal': False,
        'mismatch_percent': percent,
        'mismatch_count': mismatch_count,
        'compared_total': mismatch_total,
        'greatest_abs_diff': abs_diff,
        'abs_diff_index': abs_diff_index,
        'atol': atol,
        'greatest_rel_diff': rel_diff,
        'rel_diff_index': rel_diff_index,
        'rtol': rtol,
    }

def assert_close(a, b, tag='', assert_mismatch_percent_less_than=None, raise_exception=True, **kwargs):
    """
    Compare two tensors and check if they are close.
    
    Args:
        a: First tensor
        b: Second tensor
        mismatch_percent_threshold: If provided, tensors are considered close if mismatch_percent < threshold
        **kwargs: Additional arguments passed to torch.testing.assert_close (e.g., atol, rtol, equal_nan, etc.)
    
    Returns:
        dict with mismatch details. If tensors are close, returns dict with 'equal': True.
        If not close and exceeds threshold: raises exception
    """
    import torch
    if a is None or b is None:
        loguru.logger.error(f'Comparing None. {a=} {b=}')
        assert a is None and b is None, f'Comparing None. {a=} {b=}'

    
    try:
        torch.testing.assert_close(a, b, **kwargs)
    except Exception as e:
        # Extract mismatch information
        mismatch_info = extract_mismatch_and_diff(str(e))
        mismatch_percent = mismatch_info.get('mismatch_percent')
        
        # Check if mismatch_percent is below threshold
        if assert_mismatch_percent_less_than is not None and mismatch_percent is not None:
            if mismatch_percent < assert_mismatch_percent_less_than:
                # Consider as close even though assert_close failed
                result = mismatch_info.copy()
                result['equal'] = True
                result['within_threshold'] = True
                result['mismatch_percent_less_than'] = assert_mismatch_percent_less_than
                if tag:
                    tag_str = f'[{tag}]: '
                else:
                    tag_str = ''
                loguru.logger.info(f'{tag_str}Mismatch percent {mismatch_percent}% is less than {assert_mismatch_percent_less_than}%, considered close')
                return result
        
        # Mismatch exceeds threshold or threshold not set - raise exception
        if raise_exception:
            raise e
        return extract_mismatch_and_diff(str(e))

    return extract_mismatch_and_diff(None)

class PrecisionAligner:
    """
    A precision alignment tool for comparing outputs from different PyTorch runs with different configurations.
    
    This tool handles cases where different runs may have different world_size, rank, or parallel configurations.
    It saves tensors to a shared directory and compares them across different tasks.
    
    Example usage:
        # First run (e.g., 8 cards, sp1, dp8)
        import torch
        precision_aligner = PrecisionAligner(
            task_name='plan1', 
            tasks=['plan1', 'plan2'], 
            work_dir='./align_outputs',
            input_src='plan1'
        )
        x = torch.randn(3)
        x = precision_aligner.handle_input('input', x)
        y = x * 4
        precision_aligner.align('y', y)

        # Second run (e.g., 8 cards, sp2, dp4)
        precision_aligner = PrecisionAligner(
            task_name='plan2', 
            tasks=['plan1', 'plan2'], 
            work_dir='./align_outputs',
            input_src='plan1'
        )
        x = torch.randn(3)
        x = precision_aligner.handle_input('input', x)
        y = x * 4
        precision_aligner.align('y', y)
    """

    def __init__(self, 
                 task_name: str, 
                 tasks: List[str], 
                 work_dir: str, 
                 input_src: str, 
                 align_ranks: List[int] = None, 
                 dp_rank: int = -1, 
                 is_parallel_zero: bool = True, 
                 cache_tasks = False,
                 enable: bool = True):
        """
        Initialize the PrecisionAligner.
        
        Args:
            task_name: Name of the current task/run
            tasks: List of all task names to compare
            work_dir: Directory to save tensors for comparison
            input_src: Source task name for input tensors
            align_ranks: List of ranks to perform alignment on (default: [0])
            dp_rank: Data parallel rank (-1 means auto-detect)
            is_parallel_zero: Whether this is a parallel zero configuration
            enable: Whether to enable precision alignment
        """
        assert task_name, "task_name cannot be empty"
        assert tasks, "tasks cannot be empty"
        assert task_name in tasks, f'{task_name} not in {tasks}'
        assert work_dir, "work_dir cannot be empty"
        assert input_src, "input_src cannot be empty"

        self.task_name = task_name
        self.tasks = tasks
        self.work_dir = work_dir
        self.align_ranks = align_ranks if align_ranks is not None else [0]
        self.enable = enable
        self.dp_rank = dp_rank
        self.is_parallel_zero = is_parallel_zero
        self.input_src = input_src
        self.cache_tasks = cache_tasks
        self.cached = []

        if self.enable:
            try:
                loguru.logger.debug(f'Aligning precision for {tasks}, current run: {task_name}. Saving to {work_dir}')
                if dist.get_rank() == 0:
                    self.clear()
                self.set_seed(0)
            except Exception as e:
                print(f"Warning: Could not log alignment info: {e}")

    def get_rank(self) -> int:
        """
        Get the current rank for tensor saving/loading.
        
        Returns:
            Current rank (either dp_rank or auto-detected)
        """
        if self.dp_rank >= 0:
            return self.dp_rank

        # Try to get rank from environment variables
        rank = os.environ.get('RANK', None)
        if rank is not None:
            try:
                return int(rank)
            except ValueError:
                pass

        # Try to get rank from distributed training
        try:
            if dist.is_initialized():
                return dist.get_rank()
        except:
            pass

        # Fallback to 0
        return 0

    def should_run(self) -> bool:
        """Check if the current rank should perform alignment operations."""
        return self.get_rank() in self.align_ranks

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible results."""
        try:
            from accelerate.utils import set_seed
            set_seed(seed)
        except ImportError:
            import torch
            import random
            import numpy as np
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def get_tensor_path(self, task_name: str, tag: str) -> str:
        """Get the file path for saving/loading a tensor."""
        directory = Path(self.work_dir) / task_name
        directory.mkdir(exist_ok=True, parents=True)
        return str(directory / f"{tag}_rank{self.get_rank()}.pt")

    def get_input_tensor_path(self, tag: str) -> str:
        """Get the file path for saving/loading input tensors."""
        directory = Path(self.work_dir)
        directory.mkdir(exist_ok=True, parents=True)
        return str(directory / f"input_{tag}_rank{self.get_rank()}.pt")

    def clear(self) -> None:
        """Clear saved tensors for the current task."""
        import shutil
        directory = Path(self.work_dir) / self.task_name
        if directory.exists():
            try:
                shutil.rmtree(directory)
                loguru.logger.debug(f"Cleared directory: {directory}")
            except Exception as e:
                loguru.logger.error(f"Failed to clear directory {directory}: {e}")

    def align_with_names(self, tag: str, tensor: List, names: List[str], save_only: bool = False) -> None:
        """Align tensors with their names."""
        if not isinstance(tensor, (list, tuple)) or not isinstance(names, (list, tuple)):
            raise ValueError("tensor and names must be lists or tuples")
        if len(tensor) != len(names):
            raise ValueError("tensor and names must have the same length")

        dic = {n: t for t, n in zip(tensor, names)}
        self.align(tag, dic, save_only=save_only)

    def execute_align_tasks(self) -> None:
        """Execute alignment tasks."""
        self.cache_tasks = False
        for task in self.cached:
            print(f'Aligning {task[0]}')
            self.align(*task)
        self.cached = []
        self.cache_tasks = True

    def align(self, tag: str, tensor: Any, save_only: bool = False, pause: bool = False) -> None:
        """
        Align and compare tensors across different tasks.
        
        Args:
            tag: Tag name for the tensor
            tensor: Tensor or object to save/compare
            save_only: Only save, don't compare
            pause: Whether to pause execution on mismatch
        """
        if not self.should_run():
            return
        if not self.enable:
            return

        if self.cache_tasks:
            self.cached.append((tag, tensor, save_only, pause))
            return

        curr_path = self.get_tensor_path(self.task_name, tag)

        # Save current tensor
        if self.is_parallel_zero:
            if Path(curr_path).exists():
                loguru.logger.warning(f'{curr_path} already exists, consider if running multiple times or multiple precision checks')
            try:
                loguru.logger.debug(f'Saving tensor to {curr_path}')
                torch.save(tensor, curr_path)
            except Exception as e:
                loguru.logger.error(f'Failed to save tensor to {curr_path}: {e}')
                return

        if save_only:
            return
        if not self.is_parallel_zero:
            return

        # Compare with other tasks
        try:
            exists = all([Path(self.get_tensor_path(task, tag)).exists() for task in self.tasks])
            if exists:
                tensors = []
                for task in self.tasks:
                    try:
                        task_tensor = torch.load(
                            self.get_tensor_path(task, tag), 
                            map_location='cpu',
                        )
                        tensors.append(task_tensor)
                    except Exception as e:
                        loguru.logger.error(f'Failed to load tensor from task {task}: {e}')
                        return

                # Compare tensors
                diff_results = []
                for i, t in enumerate(tensors):
                    result = self.compare(tensor, t, f'{tag} [{self.task_name} rank vs {self.tasks[i]}]')
                    diff_results.append(result)

                all_close = all(diff_results)
                if not all_close:
                    loguru.logger.warning(f'`{tag}` mismatch detected, {self.tasks} | diff vs {self.task_name}')
                    if pause:
                        self._pause_execution()
                else:
                    loguru.logger.info(f'`{tag}` precision check passed | {self.tasks}')
            else:
                loguru.logger.info(f'Not all tasks have completed `{tag}` yet')
        except Exception as e:
            loguru.logger.error(f'Comparison failed for `{tag}`: {e}')

    def _pause_execution(self) -> None:
        """Pause execution for debugging."""
        try:
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    import pdb
                    pdb.set_trace()
                torch.distributed.barrier()
            if 'RANK' in os.environ:
                if int(os.environ['RANK']) == 0:
                    import pdb
                    pdb.set_trace()
            else:
                import pdb
                pdb.set_trace()
        except Exception as e:
            loguru.logger.warning(f'Could not pause execution: {e}')

    def handle_input(self, tag: str, obj: Any, force_not_read: bool = False) -> Any:
        """
        Handle input tensors to ensure consistency across different runs.
        
        Args:
            tag: Input tensor tag
            obj: Input object/tensor
            force_not_read: Force not to read from saved file
            
        Returns:
            Input object (either original or loaded from file)
        """
        if not self.enable:
            return obj

        self.set_seed(0)
        path = self.get_input_tensor_path(tag)

        # Use barrier only if distributed training is initialized
        try:
            if dist.is_initialized():
                dist.barrier()
        except:
            pass

        # if self.task_name == self.input_src:
        #     torch.save(obj, path)
        #     return obj

        if not Path(path).exists():
            time.sleep(1)
            loguru.logger.debug('No shared input found, saving current input')
            assert self.task_name == self.input_src, f'Current task {self.task_name} must be input source {self.input_src}'

            if self.is_parallel_zero:
                try:
                    torch.save(obj, path)
                    loguru.logger.debug(f'Saved input tensor to {path}')
                except Exception as e:
                    loguru.logger.error(f'Failed to save input tensor: {e}')
                    loguru.logger.exception(e)
            return obj
        else:
            if not force_not_read:
                loguru.logger.debug(f'Using shared input: {path}')
                try:
                    return torch.load(path, map_location='cuda', weights_only=False)
                except Exception as e:
                    loguru.logger.error(f'Failed to load input tensor: {e}')
                    return obj
            return obj

    def manual_compare(self, tag: str, tolerance: float = 1e-2) -> bool:
        """
        Manually compare tensors for a specific tag.
        
        Args:
            tag: Tag to compare
            tolerance: Tolerance threshold
            
        Returns:
            Whether all comparisons passed
        """
        try:
            tensors = []
            for task in self.tasks:
                task_tensor = torch.load(
                    self.get_tensor_path(task, tag), 
                    map_location='cpu',
                )
                tensors.append(task_tensor)

            # Compare with first tensor as reference
            comparison_results = []
            for i, t in enumerate(tensors[1:], 1):
                result = self.compare(tensors[0], t, f'{tag} [{self.task_name} vs {self.tasks[i]}]', tolerance, detailed_output=True)
                comparison_results.append(result)

            all_close = all(comparison_results)

            if not all_close:
                loguru.logger.info(f'Tensors for comparison: {[t.shape for t in tensors]}')
                loguru.logger.error(f'{tag} mismatch detected, {self.tasks}')
                self._pause_execution()
            else:
                loguru.logger.info(f'{tag} precision check passed | All task comparison results: {comparison_results}')

            return all_close

        except Exception as e:
            loguru.logger.error(f'Manual comparison failed for {tag}: {e}')
            return False

    def stop(self) -> None:
        """Stop execution for debugging."""
        if not self.enable:
            return
        self._pause_execution()

    def offline_align(self) -> None:
        """Perform offline alignment for all saved tensors."""
        try:
            for task in self.tasks:
                task_dir = Path(self.work_dir) / task
                if not task_dir.exists():
                    continue

                for file_path in task_dir.iterdir():
                    if file_path.name.startswith('input_'):
                        continue

                    # Extract tag from filename (remove _rank{rank}.pt suffix)
                    tag = file_path.stem
                    if '_rank' in tag:
                        tag = tag.rsplit('_rank', 1)[0]

                    try:
                        tensor = torch.load(file_path, map_location='cpu')
                        self.align(tag, tensor)
                    except Exception as e:
                        loguru.logger.error(f'Failed to process {file_path}: {e}')

        except Exception as e:
            loguru.logger.error(f'Offline alignment failed: {e}')

    def get_comparison_summary(self, tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Get a summary of all tensor comparisons.
        
        Args:
            tolerance: Tolerance threshold for comparisons
            
        Returns:
            Dictionary containing comparison summary
        """
        summary = {
            'total_tensors': 0,
            'passed': 0,
            'failed': 0,
            'details': {}
        }

        try:
            for task in self.tasks:
                task_dir = Path(self.work_dir) / task
                if not task_dir.exists():
                    continue

                for file_path in task_dir.iterdir():
                    if file_path.name.startswith('input_'):
                        continue

                    tag = file_path.stem
                    if '_rank' in tag:
                        tag = tag.rsplit('_rank', 1)[0]

                    if tag not in summary['details']:
                        summary['details'][tag] = {'status': 'unknown', 'tasks': []}

                    summary['details'][tag]['tasks'].append(task)
                    summary['total_tensors'] += 1

            # Check which tags have all tasks completed
            for tag, info in summary['details'].items():
                if len(info['tasks']) == len(self.tasks):
                    try:
                        passed = self.manual_compare(tag, tolerance)
                        info['status'] = 'passed' if passed else 'failed'
                        if passed:
                            summary['passed'] += 1
                        else:
                            summary['failed'] += 1
                    except Exception as e:
                        info['status'] = 'error'
                        info['error'] = str(e)
                        summary['failed'] += 1
                else:
                    info['status'] = 'incomplete'

            summary['success_rate'] = summary['passed'] / summary['total_tensors'] if summary['total_tensors'] > 0 else 0.0

        except Exception as e:
            loguru.logger.error(f'Failed to generate comparison summary: {e}')

        return summary

    def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        """
        Clean up old tensor files to save disk space.
        
        Args:
            max_age_hours: Maximum age of files in hours before deletion
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            for task in self.tasks:
                task_dir = Path(self.work_dir) / task
                if not task_dir.exists():
                    continue

                for file_path in task_dir.iterdir():
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age_seconds:
                            try:
                                file_path.unlink()
                                loguru.logger.debug(f'Deleted old file: {file_path}')
                            except Exception as e:
                                loguru.logger.warning(f'Failed to delete old file {file_path}: {e}')

        except Exception as e:
            loguru.logger.error(f'Cleanup failed: {e}')

    def compare(self, a: Any, b: Any, tag: str, tolerance: float = 1e-3, detailed_output: bool = True) -> bool:
        """
        Compare two objects with detailed analysis.
        
        Args:
            a: First object
            b: Second object  
            tag: Comparison tag
            tolerance: Tolerance for floating point comparisons
            detailed_output: Whether to output detailed information
            
        Returns:
            bool: Whether the objects are considered equal
        """
        # Type check
        if type(a) != type(b):
            error_msg = f'Comparing {tag}: Type mismatch - {type(a).__name__} vs {type(b).__name__}'
            if detailed_output:
                loguru.logger.error(error_msg)
            return False

        # PyTorch Tensor comparison
        if isinstance(a, torch.Tensor):
            return self._compare_tensor(a, b, tag, tolerance, detailed_output)

        # List comparison
        elif isinstance(a, list):
            return self._compare_sequence(a, b, tag, tolerance, detailed_output, "list")

        # Dict comparison
        elif isinstance(a, dict):
            return self._compare_dict(a, b, tag, tolerance, detailed_output)

        # Tuple comparison
        elif isinstance(a, tuple):
            return self._compare_sequence(a, b, tag, tolerance, detailed_output, "tuple")

        # Set comparison
        elif isinstance(a, set):
            return self._compare_set(a, b, tag, detailed_output)

        # NumPy array comparison
        elif hasattr(a, 'numpy') and hasattr(b, 'numpy'):
            return self._compare_numpy_array(a, b, tag, tolerance, detailed_output)

        # Basic type comparison
        else:
            if a != b:
                if detailed_output:
                    loguru.logger.error(f'Comparing {tag}: Value mismatch - {a} != {b}')
                return False
            return True


    def _compare_tensor(self, a: torch.Tensor, b: torch.Tensor, tag: str, tolerance: float, detailed_output: bool) -> bool:
        """Compare PyTorch Tensors"""
        # Basic property checks
        if a.device != b.device:
            # loguru.logger.debug(f'Comparing {tag}: Device mismatch - {a.device} vs {b.device}, converted to cpu')
            a = a.cpu()
            b = b.cpu()
            # error_msg = f'Comparing {tag}: Device mismatch - {a.device} vs {b.device}'
            # if detailed_output:
            #     loguru.logger.error(error_msg)
            # return False

        if a.dtype != b.dtype:
            # Find the lower precision dtype
            dtype_priority = {
                torch.float16: 0,
                torch.bfloat16: 1,
                torch.float32: 2,
                torch.float64: 3,
            }
            # Only consider float types for this logic
            float_types = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
            if a.dtype in float_types and b.dtype in float_types:
                lower_dtype = a.dtype if dtype_priority.get(a.dtype, 100) < dtype_priority.get(b.dtype, 100) else b.dtype
                a = a.to(lower_dtype)
                b = b.to(lower_dtype)
                loguru.logger.warning(f'Comparing {tag}: Dtype mismatch - {a.dtype} vs {b.dtype}, converted both to {lower_dtype}')
            else:
                # fallback: convert both to a's dtype
                b = b.to(a.dtype)
                loguru.logger.warning(f'Comparing {tag}: Dtype mismatch - {a.dtype} vs {b.dtype}, converted both to {a.dtype}')
            # error_msg = f'Comparing {tag}: Dtype mismatch - {a.dtype} vs {b.dtype}'
            # if detailed_output:
            #     loguru.logger.error(error_msg)
            # return False

        if a.shape != b.shape:
            error_msg = f'Comparing {tag}: Shape mismatch - {a.shape} vs {b.shape}'
            if detailed_output:
                loguru.logger.error(error_msg)
            return False


        # Non-floating point types direct comparison
        if a.dtype not in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
            return torch.equal(a, b)

        try:
            result = tensor_close(a, b)
            all_close = result.get('equal', False)
            if not all_close:
                if detailed_output:
                    msg = f"Mismatch percent: {result.get('mismatch_percent', 'N/A')}%"
                    loguru.logger.opt(depth=2).error(f'Comparing {tag}: Tensor mismatch - {msg}')
                return False
            return True
        except Exception as e:
            if detailed_output:
                loguru.logger.opt(depth=2).error(f'Comparing {tag}: Tensor mismatch - {str(e)}')
            return False

        # Floating point types use tolerance comparison
        # diff = (a - b).abs()
        # diff_avg = diff.mean().item()
        # diff_max = diff.max().item()

        # # Avoid division by zero
        # a_mean_abs = a.abs().mean().item()
        # if a_mean_abs > 1e-10:
        #     relative_diff_avg = diff_avg / a_mean_abs
        #     relative_diff_max = diff_max / a_mean_abs
        # else:
        #     relative_diff_avg = diff_avg
        #     relative_diff_max = diff_max

        # # Check if within tolerance
        # if relative_diff_avg > tolerance or relative_diff_max > tolerance * 10:
        #     if detailed_output:
        #         loguru.logger.debug(
        #             f'Comparing {tag}: \n'
        #             f'  Average relative diff: {relative_diff_avg} (tolerance: {tolerance:.2e})\n'
        #             f'  Max relative diff: {relative_diff_max} (tolerance: {tolerance*10:.2e})\n'
        #             f'  Average absolute diff: {diff_avg}\n'
        #             f'  Max absolute diff: {diff_max}'
        #         )
        #     return False

        # if detailed_output and (relative_diff_avg > tolerance * 0.1):
        #     loguru.logger.warning(
        #         f'Comparing {tag}: Tensor close to tolerance limit\n'
        #         f'  Average relative diff: {relative_diff_avg:.2e}'
        #     )

        # return True

    def _compare_sequence(self, a: Union[List, Tuple], b: Union[List, Tuple], tag: str, tolerance: float, detailed_output: bool, seq_type: str) -> bool:
        """Compare sequence types (list, tuple)"""
        if len(a) != len(b):
            error_msg = f'Comparing {tag}: {seq_type.capitalize()} length mismatch - {len(a)} vs {len(b)}'
            if detailed_output:
                loguru.logger.error(error_msg)
            return False

        results = []
        for i in range(len(a)):
            result = self.compare(a[i], b[i], f'{tag}[{i}]', tolerance, detailed_output)
            results.append(result)

        if not all(results):
            if detailed_output:
                failed_indices = [i for i, r in enumerate(results) if not r]
                loguru.logger.error(f'Comparing {tag}: {seq_type.capitalize()} elements failed at indices: {failed_indices}')
            return False

        return True

    def _compare_dict(self, a: Dict, b: Dict, tag: str, tolerance: float, detailed_output: bool) -> bool:
        """Compare dictionary types"""
        if set(a.keys()) != set(b.keys()):
            missing_keys = set(a.keys()) - set(b.keys())
            extra_keys = set(b.keys()) - set(a.keys())
            error_msg = f'Comparing {tag}: Dict key mismatch\n'
            if missing_keys:
                error_msg += f'  Missing keys: {missing_keys}\n'
            if extra_keys:
                error_msg += f'  Extra keys: {extra_keys}'

            if detailed_output:
                loguru.logger.error(error_msg)
            return False

        results = []
        for key in a.keys():
            result = self.compare(a[key], b[key], f'{tag}.{key}', tolerance, detailed_output)
            results.append(result)

        if not all(results):
            if detailed_output:
                failed_keys = [key for key, r in zip(a.keys(), results) if not r]
                loguru.logger.error(f'Comparing {tag}: Dict values failed for keys: {failed_keys}')
            return False

        return True

    def _compare_set(self, a: set, b: set, tag: str, detailed_output: bool) -> bool:
        """Compare set types"""
        if a != b:
            if detailed_output:
                missing_elements = a - b
                extra_elements = b - a
                error_msg = f'Comparing {tag}: Set mismatch\n'
                if missing_elements:
                    error_msg += f'  Missing elements: {missing_elements}\n'
                if extra_elements:
                    error_msg += f'  Extra elements: {extra_elements}'
                loguru.logger.error(error_msg)
            return False
        return True

    def _compare_numpy_array(self, a: Any, b: Any, tag: str, tolerance: float, detailed_output: bool) -> bool:
        """Compare NumPy arrays"""
        try:
            import numpy as np
            a_np = a.numpy() if hasattr(a, 'numpy') else a
            b_np = b.numpy() if hasattr(b, 'numpy') else b

            if a_np.shape != b_np.shape:
                error_msg = f'Comparing {tag}: NumPy array shape mismatch - {a_np.shape} vs {b_np.shape}'
                if detailed_output:
                    loguru.logger.error(error_msg)
                return False

            if a_np.dtype != b_np.dtype:
                error_msg = f'Comparing {tag}: NumPy array dtype mismatch - {a_np.dtype} vs {b_np.dtype}'
                if detailed_output:
                    loguru.logger.error(error_msg)
                return False

            # Floating point arrays use tolerance comparison
            if np.issubdtype(a_np.dtype, np.floating):
                if not np.allclose(a_np, b_np, rtol=tolerance, atol=tolerance):
                    diff = np.abs(a_np - b_np)
                    diff_avg = np.mean(diff)
                    diff_max = np.max(diff)
                    if detailed_output:
                        loguru.logger.error(
                            f'Comparing {tag}: NumPy array precision exceeded tolerance\n'
                            f'  Average diff: {diff_avg:.2e}\n'
                            f'  Max diff: {diff_max:.2e}'
                        )
                    return False
            else:
                if not np.array_equal(a_np, b_np):
                    if detailed_output:
                        loguru.logger.error(f'Comparing {tag}: NumPy array values not equal')
                    return False

            return True

        except ImportError:
            if detailed_output:
                loguru.logger.warning(f'Comparing {tag}: NumPy not available, falling back to basic comparison')
            return a == b

    def get_comparison_stats(self, a: Any, b: Any, tag: str) -> Dict[str, Any]:
        """
        Get detailed comparison statistics for two objects.
        
        Args:
            a: First object
            b: Second object
            tag: Comparison tag
            
        Returns:
            dict: Dictionary containing comparison statistics
        """
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return {"type": "non_tensor", "equal": a == b}

        if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
            return {
                "type": "tensor_mismatch",
                "dtype_match": a.dtype == b.dtype,
                "shape_match": a.shape == b.shape,
                "device_match": a.device == b.device,
                "a_dtype": str(a.dtype),
                "b_dtype": str(b.dtype),
                "a_shape": a.shape,
                "b_shape": b.shape,
                "a_device": str(a.device),
                "b_device": str(b.device)
            }

        if a.dtype not in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
            return {
                "type": "non_float_tensor",
                "equal": torch.equal(a, b)
            }

        # Detailed statistics for floating point tensors
        diff = (a - b).abs()
        diff_avg = diff.mean().item()
        diff_max = diff.max().item()
        diff_min = diff.min().item()
        diff_std = diff.std().item()

        a_mean_abs = a.abs().mean().item()
        a_max_abs = a.abs().max().item()

        if a_mean_abs > 1e-10:
            relative_diff_avg = diff_avg / a_mean_abs
            relative_diff_max = diff_max / a_mean_abs
            relative_diff_min = diff_min / a_mean_abs
        else:
            relative_diff_avg = relative_diff_max = relative_diff_min = float('inf')

        return {
            "type": "float_tensor",
            "absolute_diff": {
                "mean": diff_avg,
                "max": diff_max,
                "min": diff_min,
                "std": diff_std
            },
            "relative_diff": {
                "mean": relative_diff_avg,
                "max": relative_diff_max,
                "min": relative_diff_min
            },
            "reference_stats": {
                "mean_abs": a_mean_abs,
                "max_abs": a_max_abs
            },
            "shape": a.shape,
            "dtype": str(a.dtype),
            "device": str(a.device)
        }

    def batch_compare(self, objects_list: List[Tuple[Any, Any]], tag_prefix: str = "batch", tolerance: float = 1e-3, detailed_output: bool = True) -> Dict[str, Any]:
        """
        Batch compare multiple objects.
        
        Args:
            objects_list: List of object pairs to compare
            tag_prefix: Tag prefix
            tolerance: Tolerance threshold
            detailed_output: Whether to output detailed information
            
        Returns:
            dict: Dictionary containing all comparison results
        """
        results = {}

        for i, (obj_a, obj_b) in enumerate(objects_list):
            tag = f"{tag_prefix}_{i}"
            result = self.compare(obj_a, obj_b, tag, tolerance, detailed_output)
            results[tag] = {
                "passed": result,
                "stats": self.get_comparison_stats(obj_a, obj_b, tag) if isinstance(obj_a, torch.Tensor) and isinstance(obj_b, torch.Tensor) else None
            }

        # Generate summary report
        passed_count = sum(1 for r in results.values() if r["passed"])
        total_count = len(results)

        if detailed_output:
            if passed_count == total_count:
                loguru.logger.info(f"Batch comparison '{tag_prefix}': {passed_count}/{total_count} passed ✓")
            else:
                failed_tags = [tag for tag, r in results.items() if not r["passed"]]
                loguru.logger.error(f"Batch comparison '{tag_prefix}': {passed_count}/{total_count} passed ✗")
                loguru.logger.error(f"Failed comparisons: {failed_tags}")

        return {
            "summary": {
                "total": total_count,
                "passed": passed_count,
                "failed": total_count - passed_count,
                "success_rate": passed_count / total_count if total_count > 0 else 0.0
            },
            "results": results
        }




import ast
import os
from typing import Set, List
class GetImportFiles:

    def collect_imports(self, root_dir: str, greedy: bool = False) -> Set[str]:
        """
        递归收集项目中的所有导入模块路径
        :param root_dir: 项目根目录
        :param greedy: 是否解析 try/except 等复杂结构中的导入
        :return: 标准化后的模块路径集合
        """
        imports = set()

        # 遍历目录下的所有 Python 文件
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            tree = ast.parse(f.read(), filename=filepath)
                            imports.update(self._parse_imports(tree, root_dir, greedy))
                    except Exception as e:
                        print(f"解析文件 {filepath} 失败: {str(e)}")
        return imports

    def _parse_imports(self, tree: ast.AST, root_dir: str, greedy: bool) -> Set[str]:
        """解析 AST 树中的导入语句"""
        current_imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_path = self._normalize_import(alias.name, root_dir)
                    current_imports.add(module_path)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module
                if module_name is None:
                    continue  # 跳过无效的 from 导入
                # 处理相对导入
                level = node.level
                if level > 0:
                    module_path = self._resolve_relative_import(module_name, level, root_dir)
                else:
                    module_path = self._normalize_import(module_name, root_dir)
                current_imports.add(module_path)

        return current_imports

    def _normalize_import(self, module_name: str, root_dir: str) -> str:
        """将模块名转换为文件系统路径"""
        parts = module_name.split(".")
        parts[-1] = parts[-1] + '.py'
        return os.path.abspath(os.path.join(root_dir, *parts))

    def _resolve_relative_import(self, module_name: str, level: int, root_dir: str) -> str:
        """解析相对导入路径"""
        # 计算相对路径的起始目录
        parts = module_name.split(".")
        # 移除前 level 层的模块名
        base = ".".join(parts[:-level]) if level < len(parts) else ""
        # 转换为绝对路径
        return self._normalize_import(base, root_dir)


def set_debug_sys_hook():
    import sys
    from torch import distributed as dist
    from IPython.core import ultratb
    old_excpthook = sys.excepthook
    def excepthook(type, value, traceback):
        if isinstance(type, KeyboardInterrupt):
            old_excpthook(type, value, traceback)
            return
        print(type)
        print(value)
        # ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)(type, value, traceback)
        if dist.is_initialized():
            if dist.get_rank() == 0:
                ultratb.FormattedTB(mode='Plain', color_scheme='Linux', call_pdb=1)(type, value, traceback) # Plain 没那么乱
            else:
                old_excpthook(type, value, traceback)
            dist.barrier()
        else:
            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                ultratb.FormattedTB(mode='Plain', color_scheme='Linux', call_pdb=1)(type, value, traceback) # Plain 没那么乱
            else:
                old_excpthook(type, value, traceback)
    sys.excepthook = excepthook


def smart_copy(src, dst, only_link=False, copy_links=False, force_create_dir=False, verbose=True):
    """
    因为奇葩的 rsync 处理逻辑。。所以需要自己实现一个类似 rsync 的函数，从而拒绝无缘无故创建一个文件夹！然后把src放进这个文件夹！ (本来就应该是重命名行为)


    rsync file path
        assert path.parent.exist
        if path.isdir():
            -> path/file
        elif path.exists():
            Error
        else:
            path

    rsync dir path
        if path.isdir():
            -> path/dir
        elif path.exists():
            Error
        else:
            if dir.is_empty():
                path
            else:
                path/dir               [这种他妈是最诡异的]

    rsync dir/ path
        if path.isdir():
            -> path  (path/里面包含和 dir/ 一样的东西)
        elif path.exists():
            Error
        else:
            path

    """
    if '*' in str(src):
        assert str(src).endswith('/*')
        assert str(src).count('*') == 1
        src_dir = str(src).replace('/*', '')
        for f in Path(src_dir).glob('*'):
            smart_copy(f, dst, only_link=only_link, copy_links=copy_links, force_create_dir=True)
        return

    src = Path(src)
    dst = Path(dst)
    assert src.exists()
    assert dst.parent.exists()

    if force_create_dir:
        dst.mkdir(exist_ok=True)

    if copy_links:
        rsync_kwargs = '--copy-links'
    else:
        rsync_kwargs = ''

    if only_link:
        link_target = src.resolve()
        if dst.exists():
            assert dst.is_dir()
            (Path(dst) / src.name).symlink_to(link_target)
        else:
            Path(dst).symlink_to(link_target)
    else:
        def run_cmd(cmd):
            if not verbose:
                cmd = cmd + ' > /dev/null 2>&1'
            os.system(cmd)
        
        if src.is_file():
            run_cmd(f'rsync -aP {rsync_kwargs} {src} {dst}')
        else:
            if dst.exists():
                assert dst.is_dir()
                run_cmd(f'rsync -aP {rsync_kwargs} {src} {dst}')
            else: # rename mode
                run_cmd(f'rsync -aP {rsync_kwargs} {src}/ {dst}')

    def test():
        # mkdir src1 src2
        # touch src2/a src2/b src3
        # mkdir all_dst links
        smart_copy('src1', 'all_dst')
        smart_copy('src1', 'all_dst/rename_src1')
        smart_copy('src2', 'all_dst')
        smart_copy('src2', 'all_dst/rename_src2')
        smart_copy('src2/*', 'all_dst/src2_all_files')
        smart_copy('src3', 'all_dst/')
        smart_copy('src3', 'all_dst/rename_src3')

        smart_copy('src1', 'links', only_link=True)
        smart_copy('src1', 'links/rename_src1_links', only_link=True)
        smart_copy('src2', 'links', only_link=True)
        smart_copy('src2', 'links/rename_src2_links', only_link=True)
        smart_copy('src2/*', 'links/src2_all_files_links', only_link=True)
        smart_copy('src3', 'links/', only_link=True)
        smart_copy('src3', 'links/rename_src3_links', only_link=True)


        smart_copy('src1', 'links_src1', only_link=True)


        smart_copy('links', 'links_copy', copy_links=True)
        smart_copy('links', 'links_link', only_link=True)



def symbolic_links_to_real_files(save_base): # inplace replace symbolic links with real files
    for f in Path(save_base).glob('**/*'):
        if f.is_symlink():
            real_path = f.resolve()
            if real_path.exists():
                f.unlink()
                smart_copy(real_path, f, copy_links=True)
            else:
                loguru.logger.warning(f'{f} is a symbolic link but the real path {real_path} does not exist')


_timer_depth = 0
class Timer:

    def __init__(self, msg='', auto_print=True, synchronized_time=True, barrier=False):
        self.msg = msg
        if dist.is_initialized():
            self.msg = f'[Rank {dist.get_rank()}]: {msg}'
        self.auto_print = auto_print
        self.synchronized_time = synchronized_time
        self.barrier = barrier
        self.clear()
        self.entered = False
        self.__enter__()


    def __enter__(self):
        global _timer_depth
        if self.barrier:
            torch.distributed.barrier()
        if self.synchronized_time:
            torch.cuda.synchronize()
        self.start = time.time()
        if not self.entered:
            _timer_depth += 1
        self.entered = True

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        global _timer_depth
        if self.synchronized_time:
            torch.cuda.synchronize()
        self.end = time.time()
        elapsed = self.end - self.start
        self.tot += elapsed
        self.times.append(elapsed)
        if self.entered:
            _timer_depth -= 1
        if self.auto_print:
            self.print()
        self.entered = False

    def clear(self):
        self.tot = 0
        self.start = 0
        self.end = 0
        self.times = []

    def print(self):
        if len(self.times) == 0:
            loguru.logger.debug("\t" * _timer_depth + f'{self.msg} 耗时 {self.tot} 秒')
        else:
            mean_time = statistics.mean(self.times)
            std_time = statistics.stdev(self.times) if len(self.times) > 1 else 0.0
            count = len(self.times)
            loguru.logger.debug("\t" * _timer_depth + f'{self.msg} 耗时 {mean_time:.4f} ± {std_time:.4f} 秒 (共 {count} 次，总耗时 {self.tot:.4f} 秒)')

class TimerCudaEvent(Timer):
    def __enter__(self):
        torch.cuda.synchronize()
        global _timer_depth
        self.start = torch.cuda.Event(enable_timing=True)
        _timer_depth += 1

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        global _timer_depth
        self.end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        elapsed = self.start.elapsed_time(self.end)
        self.tot += elapsed
        self.times.append(elapsed)
        if self.entered:
            _timer_depth -= 1
        if self.auto_print:
            self.print()
        self.entered = False

if __name__ == '__main__':
    timer = Timer("test")
    timer2 = Timer("test2")
    for i in range(10):
        with timer:
            with timer2:
                print('inside')
            print('outside')
    timer.print()
    print(timer.get_stats())