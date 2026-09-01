import torch
from torch.distributed.tensor.placement_types import Shard
from hy_parallelism.checkpoint.checkpoint_manager import MODEL as MODEL_KEY
from hy_parallelism.engines.parallel_engine import BaseParallelEngine
from hy_parallelism.distributed.fsdp_util import apply_fsdp_checkpointing, apply_fsdp2

from .hy_base_engine import HyBaseEngine
from .fsdp_util import load_and_apply_fsdp2
from ..core.global_vars import get_args, get_parallel_state
from ..models.diffusion.leo import LeoModel, LeoDualLayer
from ..models.multimodal.hunyuan_multimodal_state import align_tensor_shape, LazyStateDict, LoadModelWrapper


class LeoEngine(HyBaseEngine):

    def fsdp_blocks(self):
        for m in self.fsdp_models:
            for block in m.layers:
                yield block
            yield m

    @staticmethod
    def valid_layer_type(model: LeoModel) -> type:
        for layer in model.layers:
            if layer is not None:
                # It should be either LeoDualLayer or CheckpointWrapper(LeoDualLayer)
                return layer.__class__
        else:
            raise ValueError("No valid block found in model layers.")

    def apply_ac(self, model: LeoModel):
        """ Apply activation checkpointing on transformer blocks before applying FSDP """
        args = get_args()

        if args.recompute_granularity is None:
            return

        if args.recompute_num_layers[0] == -1:
            recompute_percentage = 1
        else:
            recompute_percentage = args.recompute_num_layers[0] / len(model.layers)

        no_split_modules = (LeoDualLayer,)
        apply_fsdp_checkpointing(
            model,
            no_split_modules=no_split_modules,
            p=recompute_percentage,
            use_reentrant=False,
            activation_offloading=self.activation_offloading,
            activation_offload_list=[0],
        )
        if self.activation_offloading:
            from hy_parallelism.training.checkpointing import set_defer_offload
            set_defer_offload(True)

    def new_apply_fsdp(self, model: LeoModel):
        args = get_args()
        parallel_dims = self.parallel_dims
        param_dtype = torch.bfloat16 if args.bf16 else torch.float32

        model = apply_fsdp2(
            model=model,
            blocks=model.layers,
            default_fsdp_mesh=parallel_dims.default_fsdp_mesh,
            root_param_dtype=torch.float32,
            root_reduce_dtype=torch.float32,
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=args.fsdp_cast_forward_inputs,
            cpu_offload=self.cpu_offload,
            expert_fsdp_mesh=parallel_dims.expert_fsdp_mesh,
            reshard_after_forward_policy="always" if args.dp_shard <= 8 else "default",
            router_on_32=True,
            prefetch_factor=1,
            shard_placement_fn_collection=self.get_shard_placement_fn(model),
            enable_symm_mem_for_comm=args.dp_shard <= 8,
        )
        return model

    def apply_fsdp(self, model: LeoModel):
        if self.fsdp_impl == 'new':
            return self.new_apply_fsdp(model)
        else:
            return self.streaming_apply_fsdp(model)

    def streaming_apply_fsdp(self, model: LeoModel):
        args = get_args()
        parallel_dims = self.parallel_dims
        param_dtype = torch.bfloat16 if args.bf16 else torch.float32

        lazy_state_dict = LazyStateDict()
        missing_keys = []

        has_meta_param = self.has_meta(model)
        has_fsdp_plans = len(getattr(model, "fsdp_plans", [])) > 0

        if has_meta_param and has_fsdp_plans:
            # Merge all fsdp_plans into a single lazy_state_dict for streaming loading
            # when applying fully_shard.
            sources = []
            for plan in model.fsdp_plans:
                lazy_state_dict.lazy_update(plan.state_dict)
                sources.append(f"{plan.source}({plan.name})")
            print(f"Collected {len(lazy_state_dict)} parameters in lazy_state_dict for stream loading "
                  f"from following sources: {sources}", flush=True)

            # Use for stream loading checkpoints into full_tensor before fully_shard.
            def init_param_fn(name, param):
                if name in lazy_state_dict:
                    # pop to save memory
                    load_param = lazy_state_dict.pop(name)
                    load_param = align_tensor_shape(name, load_param, param)
                    param.data.copy_(load_param)
                else:
                    missing_keys.append(name)
        else:
            # No meta tensors, all parameters should be already initialized.
            init_param_fn = None

        # Determine shard placement functions
        config = model.get_config()
        if config.moe_impl == "flashinfer" or args.swap_gate_and_up:

            for name, param in model.named_parameters():
                # Assign name to param attributes for easy access
                param._name_hymm_engine_ = name

            def shard_placement_fn(param):
                # if torch.distributed.get_rank() == 0:
                #     print(f"Param ID={id(param)}, Param Shape={param.shape}", flush=True)
                param_name = param._name_hymm_engine_
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

        model = load_and_apply_fsdp2(
            model=model,
            block_type=self.valid_layer_type(model),
            default_fsdp_mesh=parallel_dims.default_fsdp_mesh,
            root_param_dtype=torch.float32,
            root_reduce_dtype=torch.float32,
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            cpu_offload=self.cpu_offload,
            expert_fsdp_mesh=parallel_dims.expert_fsdp_mesh,
            reshard_after_forward_policy="default",
            expert_gate_on_fp32=True,
            init_param_fn=init_param_fn,
            shard_placement_fn=shard_placement_fn,
        )

        if has_meta_param and has_fsdp_plans:
            unexpected_keys = list(lazy_state_dict.keys())
            # Release memory of lazy_state_dict
            lazy_state_dict.clear()
            torch.cuda.empty_cache()
            # Wait for all ranks to finish loading and then report missing/unexpected keys
            print(f"[rank {torch.distributed.get_rank()}] Waiting for all ranks to finish loading checkpoints...",)
            torch.distributed.barrier()
            rank = torch.distributed.get_rank()
            print(f"[rank {rank}] Missing keys: {missing_keys}", flush=True)
            print(f"[rank {rank}] Unexpected keys: {unexpected_keys}"
                  f"(ignored unexpected pattern: {args.ignore_unexpected_keys})", flush=True)

        return model

    def post_load_state_dict(self, default_states):
        # Post-processing after loading state dict:
        # 1. Swap gate and up weights if needed for flashinfer MoE when loading PTMv2 checkpoints
        #    For gate_and_up_proj.weight, torch implements SwiGLU with the first half being up_proj and
        #    the second half being gate_proj, while PTMv2 using TEGroupedMLP implements the opposite.
        #    Our inference engine(FlashInfer) applies the torch version, so we need to swap the two halves
        #    when loading PTMv2 weights.
        args = get_args()
        assert len(self.fsdp_models) == 1, "Only support single model for now."
        model = self.fsdp_models[0]

        if args.swap_gate_and_up:
            # Swap gate and up weights in gate_and_up_proj layer
            for layer_idx, layer in enumerate(model.layers):
                for layer_name, mlp in layer.get_mlp_layers().items():
                    if hasattr(mlp, "swap_gate_and_up_weights"):
                        if layer_idx == 0:
                            print(f"[rank {torch.distributed.get_rank()}] "
                                  f"Swaping gate and up weights in {layer_name}.gate_and_up_proj layers", flush=True)
                        mlp.swap_gate_and_up_weights()

        # Restore default states
        self.model_checkpoint_manager.states = default_states

    def apply_compile(self, model):
        from hymm.models.basic.norm_layers import HunyuanRMSNorm, LayerNormF32
        from hymm.models.diffusion import leo
        from hymm.models.basic.pos_emb_layers import apply_rope as apply_rope_native, apply_rope_qk as apply_rope_qk_native
        from hy_parallelism.models.modules.basic.pos_emb_layer import liger_apply_rope_qk
        from hy_parallelism.utils import replace_module
        from hy_parallelism.models.modules.norm_layers import FusedHunyuanRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        import os, inspect

        def get_fused_norm(name, module):
            assert isinstance(module, HunyuanRMSNorm)
            ret = FusedHunyuanRMSNorm(module.weight.shape[0], module.variance_epsilon)
            ret.weight = module.weight
            ret.variance_epsilon = module.variance_epsilon
            return ret

        replace_module(
            model, 
            is_target_module=lambda name, module: isinstance(module, (HunyuanRMSNorm)), 
            get_alternative=get_fused_norm
        )

        wrapped_sig = inspect.signature(apply_rope_qk_native)
        native_sig = inspect.signature(apply_rope_qk_native)
        if wrapped_sig != native_sig:
            raise TypeError(
                f"apply_rope_qk signature mismatch:\n  wrapped: {wrapped_sig}\n  native:  {native_sig}"
            )
        leo.apply_rope_qk = liger_apply_rope_qk

        return model
