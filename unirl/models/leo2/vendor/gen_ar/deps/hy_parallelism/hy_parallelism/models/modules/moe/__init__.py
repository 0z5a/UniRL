import os
import subprocess

from torch import distributed as dist
import loguru

try:
    from torchtitan.models.moe import MoE, MoEArgs
    from torchtitan.models.moe.moe import TokenChoiceTopKRouter
    from torchtitan.distributed.expert_parallel import ExpertParallel, DeepEPExpertParallel
    from torchtitan.models.moe.moe_deepep import DeepEPMoE
except ImportError as e:
    MoE = type(None)
    TokenChoiceTopKRouter = type(None)
    DeepEPMoE = type(None)
    ExpertParallel = type(None)
    DeepEPExpertParallel = type(None)
    pass
    # raise ImportError("torchtitan is not installed") from e
    # env = os.environ.copy()
    # env['http_proxy'] = "http://star-proxy.oa.com:3128"
    # env['https_proxy'] = "http://star-proxy.oa.com:3128"
    # if os.environ.get('LOCAL_RANK', '0') == '0':
    #     subprocess.check_call([
    #         'pip', 'install', 'torchtitan',
    #         '--force-reinstall', '--upgrade', '--no-deps', '--pre',
    #         '--index-url', 'https://download.pytorch.org/whl/nightly/cu126'
    #     ], env=env)
    # if dist.is_initialized():
    #     dist.barrier()
    # else:
    #     pass
    # from torchtitan.models.moe import MoE, MoEArgs
    # from torchtitan.distributed.expert_parallel import ExpertParallel, DeepEPExpertParallel
    # from torchtitan.models.moe.moe_deepep import DeepEPMoE

try:
    from .ptm_moe.ptm_hunyuan_moe import PTMHunYuanMoE
except Exception as e:
    loguru.logger.warning(f"Fail to import PTMHunYuanMoE. Cause: {e}")
    PTMHunYuanMoE = type(None)

try:
    from .hunyuan_moe_torch import HunYuanMoE
except Exception as e:
    loguru.logger.warning(f"Fail to import HunYuanMoE. Cause: {e}")
    HunYuanMoE = type(None)


__all__ = ['MoE', 'MoEArgs', 'DeepEPMoE', 'PTMHunYuanMoE', 'HunYuanMoE', 'ExpertParallel', 'DeepEPExpertParallel']
