# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# Reference: https://github.com/pytorch/torchtune/blob/c4835abdc78b447f72e2bcf4c2f9277582fb44fd/torchtune/training/_activation_offloading.py
# ================================================

import contextlib
from typing import Union
from warnings import warn

import psutil
import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from loguru import logger
# import torchao
# from torchao.dtypes.nf4tensor import NF4Tensor

# from torchtune.modules import TiedLinear
# from torchtune.utils import get_logger

# log = get_logger("DEBUG")


class OffloadActivationsNotrack(saved_tensors_hooks):
    """
    ReDa successfully offload activations without tracking the tensors.

    from hy_parallelism.training.activation_offload import OffloadActivations
    from hy_parallelism.training import get_activation_offload_context
    from hymm.parallelism.parallel_states import get_parallel_state
    from hy_parallelism.training.activation_offload_notrack import OffloadActivationsNotrack
    self.vae._set_gradient_checkpointing(self.vae.decoder, False)
    offload_context = get_activation_offload_context()
    offload_context.clear()
    with offload_context:
        image = self.vae_decode(latents, generator=generator)
        ref_image = self.vae_decode(ref_latents, generator=generator)
    import gc
    gc.collect()
    self.vae.decoder.eval()
    """


    def __init__(
        self,
        min_tensor_size_bytes: int = 1024,
    ) -> None:
        def get_num_bytes_tensor(x: torch.Tensor) -> int:
            # get the number of bytes in a tensor, for memory management purposes
            return (
                x.element_size() * x.nelement()
            )  # x.element_size() * x._base_storage().nbytes()

        # -------- core pack / unpack work -------- #
        def pack_tensor(activation: torch.Tensor):
            num_bytes = get_num_bytes_tensor(activation)
            if (
                activation.device.type == torch.accelerator.current_accelerator().type
                and num_bytes >= min_tensor_size_bytes
                and (
                    not isinstance(activation, torch.nn.Parameter)
                    and not (
                        hasattr(torch.nn, "Buffer")
                        and isinstance(activation, torch.nn.Buffer)
                    )
                )
            ):
                return activation.cpu()
            else:
                return activation

        def unpack_tensor(activation: torch.Tensor):
            return activation.cuda(non_blocking=True)

        super().__init__(pack_tensor, unpack_tensor)



from functools import cache
@cache
def get_activation_offload_context_notrack(*args, **kwargs):
    return OffloadActivationsNotrack(*args, **kwargs)