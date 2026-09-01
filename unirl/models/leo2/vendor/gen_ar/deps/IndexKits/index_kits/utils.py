from pathlib import Path
import numpy as np


def _arange(*args, as_tensor=False):
    try:
        import torch
    except:
        torch = None

    if torch is not None:
        data = torch.arange(*args)
        if as_tensor:
            return data
        else:
            return data.numpy()
    return np.arange(*args)


class LoadIndexError(Exception):
    def __init__(self, message, errors=[]):
        """
        LoadIndexError is raised when there is an error loading the index.

        Args:
            message (str): A general error message.
            errors (list): A list of detailed error messages.
        """
        super().__init__(message)
        self.errors = errors

    def __str__(self):
        msg = super().__str__()
        if len(self.errors) > 0:
            msg += ', '.join(self.errors)
        return msg


def get_cq_mapper(ceph_base):
    # 这个 Mapper 可以把各种各样的 ceph 盘路径转换成自己的路径
    ceph_base = {v: k for k, v in ceph_base.items()}
    CQ = ceph_base['cq']
    CQ2 = ceph_base['cq2']
    CQ3 = ceph_base['cq3']
    CQ5 = ceph_base['cq5']
    __CQ_MAPPER = {
        '/apdcephfs/share_1367250': CQ,
        '/mnt/share/cq': CQ,
        '/apdcephfs_cq2/share_1367250': CQ2,
        '/mnt/share/cq2': CQ2,
        '/apdcephfs_cq3/share_1367250': CQ3,
        '/mnt/share/cq3': CQ3,
        '/apdcephfs_cq5/share_300167803': CQ5,
        '/apdcephfs_data_cq5_2/share_300167803': CQ5,
        '/mnt/share/cq5': CQ5,
    }
    return __CQ_MAPPER


def cq_path_corrector(path, mapper):
    is_Path = isinstance(path, Path)
    if is_Path:
        path = str(path)

    assert path.startswith('/'), f'Path must be startswith /, but got {path}'
    parts = Path(path).parts
    assert parts[0] == '/', parts

    if parts[1].startswith('apdcephfs'):
        head = str(Path('/', parts[1], parts[2]))
        remain = parts[3:]
    elif parts[1].startswith('mnt'):
        head = str(Path('/', parts[1], parts[2], parts[3]))
        remain = parts[4:]
    else:
        raise ValueError(f'Unknown path: {path}. Only support paths startswith {mapper.keys()}')

    if head in mapper:
        path = Path(mapper[head], *remain)
    else:
        raise ValueError(f'Unknown path: {path}. Only support paths startswith {mapper.keys()}')

    if not is_Path:
        path = str(path)

    return path


def arrow_file_to_shadow(arrow_file, shadow_suffix, assert_exist=False):
    parts = list(Path(arrow_file).parts)
    # Add shadow_suffix to third-to-last part
    parts[-3] += shadow_suffix
    shadow_arrow_file = str(Path(*parts))
    if assert_exist:
        assert Path(shadow_arrow_file).exists(), f"Shadow arrow file not found: {shadow_arrow_file}"
    return shadow_arrow_file


def arrow_mapper(src, suffix):
    """
    Transfer /path/to/grandparent/parent/xxxxx.arrow into /path/to/grandparent{suffix}/parent/xxxxx.arrow

    Note that {suffix} will be appended to the grandparent folder.

    This function can be as the argument `arrow_mapper` of `ArrowIndexV2`, `MultiIndexV2`,
    `MultiResolutionBucketIndexV2`, and `MultiMultiResolutionBucketIndexV2`.

    Parameters
    ----------
    src: str
        Source arrow name
    suffix: str
        Suffix of new arrow name

    Returns
    -------
    new_arrow_name: str
        New arrow name
    """
    new_arrow_name = arrow_file_to_shadow(src, suffix)
    assert new_arrow_name != src, f"Arrow paths before mapper and after mapper must be different, got " \
                                  f"{new_arrow_name} vs {src}"
    return new_arrow_name


def arrow_mapper_v2(src, suffix='_caption', mapper=None):
    """
    Transfer arrows/xxx/yyyyy.arrow into arrows_caption/xxx/yyyyy.arrow
    Transfer cq2 to cq8, cq3 to cq10.

    This function can be as the argument `arrow_mapper` of `index_kits.ArrowIndexV2`,
    `index_kits.MultiIndexV2`, and `index_kits.MultiResolutionBucketIndexV2`.

    Parameters
    ----------
    src: str
        Source arrow name
    suffix: str
        Suffix of new arrow name
    mapper: dict
        Mapper to map ceph paths

    Returns
    -------
    new_arrow_name: str
        New arrow name
    """
    parts = list(Path(src).parts)
    assert 'arrows' in parts[-3], f"arrow_mapper_v2 requires `arrows` to be included in the third-to-last segment of " \
                                  f"the path. For example, '/path/to/arrows/sub_folder/00000.arrow', " \
                                  f"'/path/to/arrows_best/20240606/00001.arrow'. But got '{src}'"
    parts[-3] = parts[-3] + suffix
    new_arrow_name = str(Path(*parts))
    # --------------------------------------------------------------------------------------------
    # 由于 ceph 盘迁移, cq2 和 cq3 无法再写入了, 因此需要把 cq2 替换为 cq8, cq3 替换为 cq10
    if mapper is None:
        raise ValueError(f'`mapper` must be provided.')
    for k, v in mapper.items():
        new_arrow_name = new_arrow_name.replace(k, v)
    # --------------------------------------------------------------------------------------------
    assert new_arrow_name != src, f"{new_arrow_name} vs {src}"
    Path(new_arrow_name).parent.mkdir(parents=True, exist_ok=True)
    return new_arrow_name


def format_ceph_base(ceph_base):
    if not isinstance(ceph_base, (dict, type(None))):
        raise ValueError(f'Expected ceph_base type dict or None, got {type(ceph_base)}.')

    ceph_base = {k.rstrip('/').strip(): v.strip() for k, v in ceph_base.items()}   # remove trailing slash
    ceph_base_not_found = []
    for k in ceph_base.keys():
        if not Path(k).exists():
            ceph_base_not_found.append(f"'{k}'")
    if len(ceph_base_not_found) > 0:
        tmp_ = '\n    - '.join(ceph_base_not_found)
        raise ValueError(f""" 
    Following paths are required by the index file, but not found or not mounted in this machine:

    - {tmp_}
        """)
    return ceph_base


def round_robin_for_rank(names, rank, world_size):
    if rank < 0 or rank >= world_size:
        return []
    return [name for i, name in enumerate(names) if i % world_size == rank]


def block_for_rank(names, rank, world_size):
    if rank < 0 or rank >= world_size:
        return []

    total = len(names)
    base_chunk_size = total // world_size
    remainder = total % world_size

    if rank < remainder:
        start = rank * (base_chunk_size + 1)
        end = start + (base_chunk_size + 1)
    else:
        start = remainder * (base_chunk_size + 1) + (rank - remainder) * base_chunk_size
        end = start + base_chunk_size

    return names[start:end]


class EmptyLogger(object):
    def info(self, *args):
        pass
