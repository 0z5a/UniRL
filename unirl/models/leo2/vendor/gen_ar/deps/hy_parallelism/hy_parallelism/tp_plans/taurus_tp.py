# https://git.woa.com/leonxxli/HunyuanVideo2.0/blob/75837bcf038367a0355fd16a94db952596e0c829/hyvideo/projects/distill/parallel/taurus_tp_parallel.py

from loguru import logger

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    parallelize_module,
)


def apply_tp_deprecated(model, device_mesh, tp_group, tp_rank, tp_size):

    device_mesh = device_mesh["tp"]
    for layer_id, double_block in enumerate(model.double_blocks):
        tp_plan = {
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
        double_block.heads_num = double_block.heads_num // tp_size
        parallelize_module(double_block, device_mesh, tp_plan)

    logger.info(f"DoubleBlocks tp is initialized ", rank0=True)

    for layer_id, single_block in enumerate(model.single_blocks):
        tp_plan = {
            "linear1_q": ColwiseParallel(),
            "linear1_k": ColwiseParallel(),
            "linear1_v": ColwiseParallel(),
            "linear1_mlp": ColwiseParallel(),
            "linear2": PrepareModuleInput(
                input_layouts=(Shard(-1), Shard(-1)), desired_input_layouts=(Replicate(), Replicate())
            ),
            "linear2.fc": ColwiseParallel(output_layouts=Replicate()),
        }

        single_block.heads_num = single_block.heads_num // tp_size
        parallelize_module(single_block, device_mesh, tp_plan)
    logger.info(f"SingleBlocks tp is initialized ", rank0=True)
    #! TODO: TokenRefiner TP : Parallelizing the attention layers introduces errors that lead to the loss of important semantic information in the generated images.

    # for layer_id, token_refiner_block in enumerate(model.txt_in.individual_token_refiner.blocks):
    #     tp_plan = {
    #     # "self_attn_qkv": ColwiseParallel(),
    #     # "self_attn_proj": RowwiseParallel(),
    #     "mlp.fc1": ColwiseParallel(),
    #     "mlp.fc2": RowwiseParallel(),
    #     }
    #     token_refiner_block.layer_id = layer_id
    #     parallelize_module(token_refiner_block, device_mesh, tp_plan)
    # logger.info(f"TokenRefiners TP is initialized ", rank0=True)

def apply_tp(self, model):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module, PrepareModuleInput
    from torch.distributed._tensor import Shard, Replicate
    from hy_parallelism.parallel_states import get_parallel_state

    device_mesh = get_parallel_state().tp_mesh
    tp_size = device_mesh.size()

    for layer_id, double_block in enumerate(model.double_blocks):
        tp_plan = {
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
        double_block.heads_num = double_block.heads_num // tp_size
        parallelize_module(double_block, device_mesh, tp_plan)

    logger.info(f"DoubleBlocks tp is initialized ", rank0=True)

    for layer_id, single_block in enumerate(model.single_blocks):
        tp_plan = {
            "linear1_q": ColwiseParallel(),
            "linear1_k": ColwiseParallel(),
            "linear1_v": ColwiseParallel(),
            "linear1_mlp": ColwiseParallel(),
            "linear2": PrepareModuleInput(
                input_layouts=(Shard(-1), Shard(-1)), desired_input_layouts=(Replicate(), Replicate())
            ),
            "linear2.fc": ColwiseParallel(output_layouts=Replicate()),
        }

        single_block.heads_num = single_block.heads_num // tp_size
        parallelize_module(single_block, device_mesh, tp_plan)
    logger.info(f"SingleBlocks tp is initialized ", rank0=True)
    raise NotImplementedError("TP precision is not checked yet.")