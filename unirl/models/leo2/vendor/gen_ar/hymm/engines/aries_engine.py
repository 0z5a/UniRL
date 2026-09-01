import torch
from hy_parallelism.engines.parallel_engine import BaseParallelEngine
from hy_parallelism.distributed.fsdp_util import apply_fsdp_checkpointing

from .fsdp_util import load_and_apply_fsdp2
from ..core.global_vars import get_args
from ..models.diffusion.aries import Aries, DoubleStreamBlock, SingleStreamBlock


class AriesEngine(BaseParallelEngine):

    @property
    def monitor(self):
        if not hasattr(self, '_monitor'):
            return None
        return self._monitor

    @monitor.setter
    def monitor(self, monitor):
        self._monitor = monitor

    def fsdp_blocks(self):
        for m in self.fsdp_models:
            for block in m.double_blocks:
                yield block
            for block in m.single_blocks:
                yield block
            yield m

    @staticmethod
    def valid_layer_type(model: Aries) -> type:
        for layer in model.model.layers:
            if layer is not None:
                # It should be either HunyuanMultimodalLayer or CheckpointWrapper(HunyuanMultimodalLayer)
                return layer.__class__
        else:
            raise ValueError("No valid block found in model layers.")

    def apply_ac(self, model: Aries):
        """ Apply activation checkpointing on transformer blocks before applying FSDP """
        args = get_args()

        if args.recompute_granularity is None:
            return

        if args.recompute_num_layers[0] == -1:
            recompute_percentage = 1
        else:
            recompute_percentage = args.recompute_num_layers[0] / (model.depth_double_blocks + model.depth_single_blocks)

        apply_fsdp_checkpointing(
            model,
            no_split_modules=(DoubleStreamBlock, SingleStreamBlock),
            p=recompute_percentage,
            use_reentrant=True,
        )

    def apply_fsdp(self, model: Aries):
        args = get_args()
        parallel_dims = self.parallel_dims
        param_dtype = torch.bfloat16 if args.bf16 else torch.float32

        model = load_and_apply_fsdp2(
            model=model,
            block_type=(DoubleStreamBlock, SingleStreamBlock),
            default_fsdp_mesh=parallel_dims.default_fsdp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            cpu_offload=self.cpu_offload,
            reshard_after_forward_policy="default",
            expert_gate_on_fp32=False,
            vit_on_fp32=args.vit_precision,
        )

        return model

    def pre_load_state_dict(self):
        return

    def post_load_state_dict(self, default_states):
        return
