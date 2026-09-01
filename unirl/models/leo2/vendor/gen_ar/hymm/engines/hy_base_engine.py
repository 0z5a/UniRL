import torch
from torch.distributed.tensor.placement_types import Shard
from hy_parallelism.checkpoint.checkpoint_manager import MODEL as MODEL_KEY
from hy_parallelism.engines.parallel_engine import BaseParallelEngine
from hy_parallelism.distributed.fsdp_util import apply_fsdp_checkpointing, apply_fsdp2

from .fsdp_util import load_and_apply_fsdp2
from ..core.global_vars import get_args, get_parallel_state
from ..models.diffusion.leo import LeoModel, LeoDualLayer
from ..models.multimodal.hunyuan_multimodal_state import HunyuanMultimodalState, align_tensor_shape, LazyStateDict, LoadModelWrapper


class HyBaseEngine(BaseParallelEngine):

    def __init__(self, model, *args, **kwargs):
        # For testing: keep both implementations for now. Prefer new_apply_fsdp,
        # but if you encounter issues, you can easily switch back to streaming_apply_fsdp.
        # TODO: remove this after testing.
        self.fsdp_impl = get_args().fsdp_impl
        if self.fsdp_impl == 'new':
            # TODO: 让 flash infer MOE 创建的时候就把参数正确创建
            #       因为未来不需要先 hf load 好 deepseek 的 checkpoint 再带着参数值 materialize
            for module in self.recursive_module_generator_buttom_up(model):
                setattr(module, "_materialize_and_init_state", True)
        super().__init__(model, *args, **kwargs)
        if self.fsdp_impl == 'new':
            self.maybe_load_fsdp_plan(self.module)


    def param_init_fn(self, model, default_generator=None):
        pass

    @property
    def monitor(self):
        if not hasattr(self, '_monitor'):
            return None
        return self._monitor

    @monitor.setter
    def monitor(self, monitor):
        self._monitor = monitor


    def maybe_load_fsdp_plan(self, model, default_generator=None):
        """
        fsdp, to_emtpy, reset_parameter 之后的最后一步初始化
        是为了加载 hf 或者 bin 才有设计这一步，dcp 的参数转换写在 LoadModelWrapper 中
        """

        lazy_state_dict = LazyStateDict()
        missing_keys = []

        has_fsdp_plans = len(getattr(model, "fsdp_plans", [])) > 0

        if has_fsdp_plans:
            # Merge all fsdp_plans into a single lazy_state_dict for streaming loading
            # when applying fully_shard.
            sources = []
            for plan in model.fsdp_plans:
                lazy_state_dict.lazy_update(plan.state_dict)
                sources.append(f"{plan.source}({plan.name})")
            print(f"Collected {len(lazy_state_dict)} parameters in lazy_state_dict for stream loading "
                  f"from following sources: {sources}", flush=True)

            # NOTE(kevinkhwu):
            #   这里调用这个只是最后的兜底操作，一般不会起作用
            #   因为 state_dict_from_bin 和 state_dict_from_hf 已经对扩此表，moe 等做了参数调整
            #   lazy_state_dict 已经是一个 shape_align 的 sd 了
            #   这里面按道理不应该出现 shape mismatch
            # TODO(kevinkhwu):
            #   重新设计这类实现，重构 collect_load_plans 和 state_dict_from_bin/state_dict_from_hf
            from hymm.checkpoint.state_dict_mapping import maybe_extend_sd_vocab, maybe_convert_deepseek_moe_to_flashinfer
            lazy_state_dict = maybe_extend_sd_vocab(lazy_state_dict, model.state_dict())
            if model.get_config().moe_impl == "flashinfer":
                lazy_state_dict = maybe_convert_deepseek_moe_to_flashinfer(lazy_state_dict, self.args, model.get_config())

            missing_keys, unexpected_keys = self.load_state_dict(lazy_state_dict, strict=False, is_non_standard_sharded_sd=True)

            from hy_parallelism.utils import get_missing_unexpected_str
            print(get_missing_unexpected_str(missing_keys, unexpected_keys, tag='FSDP Plan'))
            print(f"(ignored unexpected pattern: {self.args.ignore_unexpected_keys})", flush=True)


    def get_shard_placement_fn(self, model: HunyuanMultimodalState):
        """flashinfer 和 swap_gate_and_up 都需要特殊的 shard_placement_fn 实现。"""

        args = get_args()
        if model.get_config().moe_impl == "flashinfer" or args.swap_gate_and_up:
            self._tag_param_name_to_params(model)

            def shard_placement_fn(param):
                param_name = self._get_param_name_from_param(param)
                if "gate_and_up" in param_name or "expert_down_weights" in param_name:
                    # 1. 'expert_gate_and_up_weights' and 'expert_down_weights' use combined weights for
                    #    flashinfer while separated experts weights in PureTorch dcp on dim(0).
                    # 2. 'shared_mlp.gate_and_up_proj.weight' and 'expert_gate_and_up_weights' will be
                    #    applied swap_gate_and_up_weights during loading PTMv2 dcp on dim(-2).
                    # Therefore, dim(-1) is the only safe dim to shard those parameters for DTensor.
                    return Shard(-1)
                else:
                    return None
        else:
            shard_placement_fn = None
        return shard_placement_fn


    def is_moe_router(self, module_full_name, module):
        if isinstance(module_full_name, str) and \
            (
                module_full_name.endswith('.router') or 
                (module_full_name.endswith('.gate') and 'router' not in module_full_name) # puretorch gate, filter out router.gate (submodule of titan router)
            ):
            return True
        # TODO: Determine if the module is a router module by checking the fqn
        return False

    def pre_load_state_dict(self):
        args = get_args()
        p_state = get_parallel_state()

        old_states = self.model_checkpoint_manager.states.copy()

        if p_state.ep_size == 1:
            # Handle key-mapping or parameter rearrangement
            load_wrapper = LoadModelWrapper(self.fsdp_models)
            if args.load_remove_prefix:
                self.model_checkpoint_manager.states = load_wrapper.state_dict()
            else:
                self.model_checkpoint_manager.states[MODEL_KEY] = load_wrapper
        else:
            assert self.module.get_config().moe_impl != "flashinfer", "Loading dcp checkpoint with flash_infer does not support expert parallel"
            assert len(self.module.get_key_mapping()) == 0, "Loading dcp checkpoint with key_mapping does not support expert parallel"
            assert not args.load_remove_prefix, "Loading dcp checkpoint with load_remove_prefix does not support expert parallel"

        return old_states

    def post_load_state_dict(self, default_states):
        pass