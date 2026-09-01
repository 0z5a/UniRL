# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    parallelize_module,
)


def apply_tp(model, tp_mesh):
    parallelize_module(
        model,
        tp_mesh,
        {},
    )

    for layer_id, double_block in enumerate(model.double_blocks):
        layer_plan = {
            "img_attn_q": ColwiseParallel(),
            "img_attn_k": ColwiseParallel(),
            "img_attn_v": ColwiseParallel(),
            "img_attn_proj": RowwiseParallel(),
            "img_mlp.fc1": ColwiseParallel(),
            "img_mlp.fc2": RowwiseParallel(),
            "txt_attn_q": ColwiseParallel(),
            "txt_attn_k": ColwiseParallel(),
            "txt_attn_v": ColwiseParallel(),
            "txt_attn_proj": RowwiseParallel(),
            "txt_mlp.fc1": ColwiseParallel(),
            "txt_mlp.fc2": RowwiseParallel(),
        }
        double_block.heads_num = double_block.heads_num // tp_mesh.size()

        parallelize_module(
            module=double_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    for layer_id, single_block in enumerate(model.single_blocks):
        layer_plan = {
            "linear1_q": ColwiseParallel(),
            "linear1_k": ColwiseParallel(),
            "linear1_v": ColwiseParallel(),
            "linear1_mlp": ColwiseParallel(),
            "linear2": PrepareModuleInput(
                input_layouts=(Shard(-1), Shard(-1)), desired_input_layouts=(Replicate(), Replicate())
            ),
            "linear2.fc": ColwiseParallel(output_layouts=Replicate()),
        }
        single_block.heads_num = single_block.heads_num // tp_mesh.size()

        parallelize_module(
            module=single_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )
