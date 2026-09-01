from argparse import Namespace
from dataclasses import dataclass
from typing import Optional

import torch
from megatron.core.transformer import MegatronModule, TransformerConfig

from .hunyuan_multimodal import HunyuanMultimodalBase
from .hunyuan_multimodal_config import HunyuanMultimodalConfig


@dataclass
class HunyuanMultimodalMCoreConfig(HunyuanMultimodalConfig, TransformerConfig):
    # Training with mcore requires a TransformerConfig for model.
    pass


class HunyuanMultimodalMCore(HunyuanMultimodalBase, MegatronModule):
    """ A wrapper of Transfusion model to be compatible with mcore framework. """
    def __init__(
            self,
            args: Namespace,
            config: HunyuanMultimodalMCoreConfig,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        super().__init__(config)
        self.__post_init__(args, config, dtype, device)

    # ===============================
    #     mcore required methods
    # ===============================

    def set_input_tensor(self, input_tensor) -> None:
        self.input_tensor = input_tensor
