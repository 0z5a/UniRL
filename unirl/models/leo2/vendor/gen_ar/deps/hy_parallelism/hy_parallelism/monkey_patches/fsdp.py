from packaging import version
import torch
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch import distributed as dist


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



def patch_record_post_forward():
    # 不然，L6.gate 在 backward 时会 prefetch 已经 backward 完并 reshard 的 L7.expert
    if version.parse(torch.__version__) >= version.parse('2.10.0rc6'):
        return
    from hy_parallelism.utils import is_recomputing
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    original_record_post_forward = FSDPParamGroup._record_post_forward
    def _record_post_forward(self):
        if not is_recomputing():
            return original_record_post_forward(self)
        return None
    FSDPParamGroup._record_post_forward = _record_post_forward


def patch_skipped_unshard_in_dual_stream_ac():
    """
    这个bug会导致其中一个stream recompute 的时候跳过 unshard, 从何出现 DTensor 和 Tensor 的 device mismatch
    # With nested FSDP and multiple forward passes before backward,
    # the params might have been resharded by a previous post_backward.
    # We need to ensure params are unsharded for AC recomputation.

    最小复现：

    import hy_parallelism
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.distributed._composable import checkpoint
    from torch.distributed.fsdp import fully_shard

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 8)
        def forward(self, x):
            return self.fc2(self.fc1(x).relu())

    if __name__ == "__main__":
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        m = Block().cuda()
        checkpoint(m)
        fully_shard(m.fc1)
        x = torch.randn(2, 8, device="cuda", requires_grad=True)
        (m(x) * m(x)).sum().backward()
    """
    from typing import Any
    from torch.distributed.fsdp._fully_shard._fsdp_state import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_common import _cast_fp_tensor
    from torch.distributed.fsdp._fully_shard import _fsdp_state
    from torch.distributed.utils import _apply_to_tensors, _to_kwargs
    import functools
    from torch import nn

    original_pre_forward = _fsdp_state.FSDPState._pre_forward
    original_post_forward = _fsdp_state.FSDPState._post_forward

    def _pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self._training_state == TrainingState.PRE_BACKWARD:
            # With nested FSDP and multiple forward passes before backward,
            # the params might have been resharded by a previous post_backward.
            # We need to ensure params are unsharded for AC recomputation.
            if self._fsdp_param_group is not None and not self._fsdp_param_group.is_unsharded:
                self._fsdp_param_group.unshard()
                self._fsdp_param_group.wait_for_unshard()
            with torch.profiler.record_function("FSDP::cast_forward_inputs"):
                cast_fn = functools.partial(
                    _cast_fp_tensor, self._mp_policy.param_dtype
                )
                args, kwargs = (
                    _apply_to_tensors(cast_fn, args),
                    _apply_to_tensors(cast_fn, kwargs),
                )
        return original_pre_forward(self, module, args, kwargs)


    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        # release mistargeted prefetch
        for sub_module in module.modules():
            if sub_module is not module and isinstance(sub_module, FSDPModule):
                # Copied from: sub_module._get_fsdp_state()._fsdp_param_group.finalize_backward()
                pg = sub_module._get_fsdp_state()._fsdp_param_group
                if pg._all_gather_result is not None:
                    # If there was a mistargeted unshard without a corresponding wait,
                    # then we wait here and clear the unshard
                    if (event := pg._all_gather_result.all_gather_event) is not None:
                        torch.accelerator.current_stream().wait_event(event)
                    work = pg._all_gather_result.all_gather_work
                    if isinstance(work, dist.distributed_c10d.Work):
                        work.wait()
                    pg._all_gather_result = None
        return original_post_forward(self, module, input, output)


    if version.parse(torch.__version__) <= version.parse('2.11.0'):
        _fsdp_state.FSDPState._pre_forward = _pre_forward
        _fsdp_state.FSDPState._post_forward = _post_forward