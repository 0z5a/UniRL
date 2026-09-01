from packaging import version
import torch


def patch_reshard_after_forward():
    from typing import cast
    if version.parse(torch.__version__) >= version.parse('2.6.0'):
        from torch import nn
        from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
        from torch.distributed.fsdp._fully_shard._fully_shard import _get_post_forward_mesh_info
    else:
        from torch.distributed._composable.fsdp import (
            FSDPModule, 
        )
        from torch.distributed._composable.fsdp._fsdp_init import _get_post_forward_mesh_info


    def set_reshard_after_forward(
        self, reshard_after_forward: bool, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after forward. This can be
        used to change the ``reshard_after_forward`` FSDP arg at runtime. For
        example, this can be used to set the FSDP root module's value to
        ``True`` (since it is otherwise specially set to ``False``), or it can
        set an FSDP module's value to ``False`` for running evals and set back
        to ``True`` for training.

        Args:
            reshard_after_forward (bool): Whether to reshard parameters after
                forward.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.post_forward_mesh_info = (
                        _get_post_forward_mesh_info(
                            reshard_after_forward, fsdp_param_group.mesh_info
                        )
                    )

    if not hasattr(FSDPModule, 'set_reshard_after_forward'):
        FSDPModule.set_reshard_after_forward = set_reshard_after_forward