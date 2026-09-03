from packaging import version
import torch
from torch.nn import utils as nn_utils
import loguru

from hy_parallelism.utils import _get_total_norm, _clip_grads_with_norm_

def patch_torch_nn_utils_grad_norm():
    # pytorch < 2.6.0 does not have get_total_norm
    if not hasattr(nn_utils, 'get_total_norm'):
        nn_utils.get_total_norm = _get_total_norm
    if not hasattr(nn_utils, 'clip_grads_with_norm_'):
        nn_utils.clip_grads_with_norm_ = _clip_grads_with_norm_

def patch_torch_grouped_mm():
    if not hasattr(torch, '_grouped_mm'):
        try:
            from hy_parallelism.triton.moe.grouped_gemm_v1 import triton_grouped_mm
        except Exception as e:
            # Could raise error when running without GPU
            loguru.logger.warning(f"Fail to import triton_grouped_mm. Cause: {e}")
            return
        loguru.logger.warning(
            f'You are using an old PyTorch version ({torch.__version__}). '
            'Patching torch._grouped_mm with triton_grouped_mm, which is an experimental implementation.'
            'Please consider upgrading to PyTorch 2.8.1 or later. '
            'This warning can be ignored if you are not using the MOE with grouped GEMM implementation.'
        )
        torch._grouped_mm = triton_grouped_mm


def reproduce_device_mesh_hash_collision():
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.tensor.device_mesh import _mesh_resources

    # Slicing again will overwrite child_to_root_mapping. since the hash of the two submeshes are identical.
    slice_again = True

    mesh1 = init_device_mesh(device_type='cuda', mesh_shape=(2, 2,), mesh_dim_names=('cp', 'tp',))
    mesh2 = init_device_mesh(device_type='cuda', mesh_shape=(2, 2,), mesh_dim_names=('cp', 'tp',))

    tp_mesh1 = mesh1['tp']
    if slice_again:
        tp_mesh2 = mesh2['tp']

    assert _mesh_resources.get_root_mesh(tp_mesh1) is mesh1

def patch_torch_device_mesh_hash_collision():
    # https://github.com/pytorch/pytorch/issues/173789
    if version.parse(torch.__version__) <= version.parse("2.20.1"):
        from torch.distributed.tensor.device_mesh import DeviceMesh
        def _DeviceMesh__hash__(self):
            from hy_parallelism.parallel_states import _PARALLEL_STATE_KEY
            # lazily compute hash
            self._hash = getattr(self, "_hash", None)
            if not self._hash:
                # loguru.logger.opt(depth=3).warning(f'换了 hash 用 {repr(_PARALLEL_STATE_KEY)}')
                self._hash = hash(
                    (
                        self._flatten_mesh_list,
                        self.mesh.shape,
                        self.device_type,
                        self.mesh_dim_names,
                        self._thread_id,
                        _PARALLEL_STATE_KEY
                    )
                )
            return self._hash
        DeviceMesh.__hash__ = _DeviceMesh__hash__


def patch_pipeline_schedule():
    """
    Removing raise for GPipe

    https://github.com/pytorch/pytorch/issues/171312
    """

    from packaging import version

    import torch
    from torch.distributed.pipelining.schedules import PipelineScheduleSingle

    def new_pipeline_schedule_single_init(
        self,
        stage,
        n_microbatches,
        loss_fn=None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_merge_spec=None,
        scale_grads=True,
    ):
        # Init parent - use explicit super() call to work with monkey-patching
        super(PipelineScheduleSingle, self).__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        self._stage_initialized = False

        self.pipeline_order = (
            self._get_pipeline_order()
        )

    # old version works fine
    if version.parse(torch.__version__) > version.parse('2.7.0') and version.parse(torch.__version__) <= version.parse('2.9.1'):
        PipelineScheduleSingle.__init__ = new_pipeline_schedule_single_init


def patch_pick_load_for_old_python():
    # 支持旧 python 读取 3.13 的 pickle 文件
    import os

    import pickle
    original_pickle_load = pickle.load
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                ret = super().find_class(module, name)
                return ret
            except:
                if module == "pathlib._local":
                    from pathlib import Path
                    return Path
                raise

    def pickle_load(file, *, fix_imports=True, encoding="ASCII", errors="strict"):
        try:
            ret =  original_pickle_load(file)
            return ret
        except (BaseException,  ModuleNotFoundError) as e:
            import io
            file.seek(0)
            data = file.read()
            return _CompatUnpickler(io.BytesIO(data), fix_imports=fix_imports, encoding=encoding, errors=errors).load()

    pickle.load = pickle_load

def patch_dcp_error_pickle_code_object_for_python313():
    import sys
    import traceback
    import torch.distributed.checkpoint.utils as dcp_utils

    if sys.version_info >= (3, 13):
        def _wrap_exception_py313(
            exc: BaseException
        ) -> tuple[BaseException, traceback.StackSummary]:

            frames = traceback.extract_tb(exc.__traceback__)
            safe_summary = traceback.StackSummary.from_list([
                (f.filename, f.lineno, f.name, f.line or '') for f in frames  # type: ignore
            ])

            return (exc.with_traceback(None), safe_summary)

        dcp_utils._wrap_exception = _wrap_exception_py313  # type: ignore


def patch_dcp_dist_wrapper():
    """
    網卡容納嘅 RDMA 連接數系有限嘅，需要避免大規模 P2P 連接。

    Upstream ``reduce_scatter`` is implemented as ``gather_object`` + ``scatter_object``
    (P2P). At large scale the coordinator must open connections to every rank, which
    can trigger NCCL errors or OOM.

    NCCL_PXN_DISABLE=0
    NCCL_IB_QPS_PER_CONNECTION=1

    !532 !1104
    """
    import os
    from typing import Callable, Optional, TypeVar, Union, cast

    if os.environ.get("HY_PARALLELISM_NO_PATCH_DCP_DIST_WRAPPER", "0") == "1":
        return

    import torch.distributed.checkpoint.utils as dcp_utils
    from torch.distributed.checkpoint.api import CheckpointException, WRAPPED_EXCEPTION

    _DistWrapper = dcp_utils._DistWrapper
    _get_failure_dict = dcp_utils._get_failure_dict
    T = TypeVar("T")
    R = TypeVar("R")

    def reduce_scatter(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[list[T]], list[R]],
    ) -> R:
        # Look up at call time so patch_dcp_error_pickle_code_object_for_python313
        # (and any later wrap patches) remain effective.
        wrap_exception = dcp_utils._wrap_exception

        local_data: Union[WRAPPED_EXCEPTION, T]
        try:
            local_data = map_fun()
        except BaseException as e:  # noqa: B036
            local_data = wrap_exception(e)

        all_data = self.all_gather_object(local_data)

        all_results: Optional[list[Union[R, CheckpointException]]] = None
        if self.is_coordinator:
            node_failures = _get_failure_dict(all_data)

            if len(node_failures) == 0:
                try:
                    all_results = cast(
                        list[Union[R, CheckpointException]],
                        reduce_fun(cast(list[T], all_data)),
                    )
                except BaseException as e:  # noqa: B036
                    node_failures[self.rank] = wrap_exception(e)

            if len(node_failures) > 0:
                all_results = [CheckpointException(step, node_failures)] * self.get_world_size()

        broadcast_results = self.broadcast_object(all_results)
        assert broadcast_results is not None
        result = broadcast_results[self.rank]

        if isinstance(result, CheckpointException):
            raise result
        return result

    def gather_object(self, object):
        # Avoid P2P gather used by all_reduce / older reduce_scatter paths.
        return self.all_gather_object(object)

    _DistWrapper.reduce_scatter = reduce_scatter  # type: ignore[method-assign]
    _DistWrapper.gather_object = gather_object  # type: ignore[method-assign]


def patch_checkpoint_wrapper():
    """ 让推理时 不要执行任何 checkpointing 相关的操作 """

    from torch.distributed.algorithms._checkpoint import checkpoint_wrapper
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

    class CheckpointWrapper(checkpoint_wrapper.CheckpointWrapper):
        def forward(self, *args, **kwargs):
            # is_grad_enabled = False
            # if not torch.is_grad_enabled():
            #     is_grad_enabled = False
            # else:
            #     if self.training:
            #         is_grad_enabled = True
            #     else:
            #         is_grad_enabled = next(self.parameters()).requires_grad

            #     # FIXME: 目前先只检查一层，handle不了把tensor藏得很深传入forward的场景
            #     if not is_grad_enabled:
            #         try:
            #             for k in args:
            #                 if isinstance(k, torch.Tensor) and k.requires_grad:
            #                     is_grad_enabled = True
            #                     break
            #             for k, v in kwargs.items():
            #                 if isinstance(v, torch.Tensor) and v.requires_grad:
            #                     is_grad_enabled = True
            #                     break
            #         except:
            #             pass

            import os
            if os.getenv('SKIP_CHECKPOINTING', '0') == '1' or not torch.is_grad_enabled():
                return self._checkpoint_wrapped_module(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    checkpoint_wrapper.CheckpointWrapper = CheckpointWrapper