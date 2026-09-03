# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

import os

default_env = {
    'EP_SUPPRESS_NCCL_CHECK': '1',
    'EP_NCCL_ROOT_DIR': '/usr/local/tccl',
    'NCCL_GIN_TYPE': '3',
    'NCCL_DMABUF_ENABLE': '1',
    'NCCL_GDRCOPY_ENABLE': '1',
    'NCCL_GIN_ENABLE': '1',
}
for key, value in default_env.items():
    if key not in os.environ:
        os.environ[key] = value

def _use_deepep_v2() -> bool:
    try:
        import deep_ep
    except ImportError as e:
        raise ImportError(
            "deep_ep is required for moe_parallel_deepep. "
            "Install from https://github.com/deepseek-ai/DeepEP."
            "For Taiji server, please use a correct mirror or contact kevinkhwu."
        ) from e

    if hasattr(deep_ep, "ElasticBuffer"):
        return True

    version = getattr(deep_ep, "__version__", "0.0.0")
    try:
        return int(version.split(".", maxsplit=1)[0]) >= 2
    except ValueError:
        return False

if _use_deepep_v2():
    version = 'v2'
    from .moe_parallel_deepep_v2 import preprocess, token_pre_all2all, tokens_post_all2all, set_low_latency
else:
    version = 'v1'
    from .moe_parallel_deepep_v1 import preprocess, token_pre_all2all, tokens_post_all2all, set_low_latency
