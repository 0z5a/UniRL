import argparse
from functools import partial

import torch

from megatron.core import parallel_state
from megatron.training import get_args
from ..core.data_provider import (
    DatasetsProvider,
    prepare_model_t2i_inputs,
    prepare_model_lm_inputs,
    prepare_model_mmu_inputs,
)
from ..core.extra_model_provider import (
    build_scalar_state,
    build_vae,
    build_denoiser,
    build_tkwrapper,
)
from ..core.global_vars import get_mm_state
from ..utils.helpers import multi_pattern_match, print_args
from ..utils.torch_utils import set_manual_seed


def model_provider(pre_process=True, post_process=True):
    """ Build the model. """
    from hymm.models import build_model

    _ = pre_process
    _ = post_process

    args = get_args()
    dp_rank = parallel_state.get_data_parallel_rank()

    # When training torch model using ptm v2, ensure model structure ends with MCore
    if not args.model_structure.endswith("MCore"):
        args.model_structure += "MCore"

    # initialize scalar states, denoiser, vae, tkwrapper as global vars
    build_scalar_state()
    build_denoiser()
    build_vae()
    build_tkwrapper()

    # build and initialize main model
    set_manual_seed(args.seed)
    model, model_config = build_model(args)
    set_manual_seed(args.seed + dp_rank)

    # Print model config
    if args.rank == 0:
        print_args("model config", model_config)

    return model


def loss_func(loss_dict: dict[str, torch.Tensor], output_tensor: torch.Tensor):
    _ = output_tensor
    mm_state = get_mm_state()
    loss = loss_dict['loss'].float()

    loss_names = ["loss", "image_loss", "text_loss"] + [
        f"{dataset_tag}_text_loss" for dataset_tag in mm_state.all_dataset_keys
    ] + [
        f"{dataset_tag}_image_loss" for dataset_tag in mm_state.all_dataset_keys
    ]
    loss_values = []
    loss_counts = []
    for name in loss_names:
        if name in loss_dict:
            loss_values.append(loss_dict[name].detach().clone().float().view(1))
            loss_counts.append(torch.tensor([1.0], device=loss.device))
        else:
            loss_values.append(torch.tensor([0.0], device=loss.device))
            loss_counts.append(torch.tensor([0.0], device=loss.device))

    loss_and_count = torch.cat(loss_values + loss_counts)
    torch.distributed.all_reduce(
        loss_and_count,
        group=parallel_state.get_data_parallel_group(with_context_parallel=False)
    )
    sum_losses, sum_counts = torch.split(loss_and_count, len(loss_names))
    sum_counts = sum_counts.clamp(min=1.0)    # avoid division by zero

    loss_reduced = {}
    for name, log_loss, count in zip(loss_names, sum_losses, sum_counts):
        # reporting loss must concat denominator, see the process of losses_reduced in train_step
        loss_reduced[name] = torch.cat([log_loss.view(1), count.view(1)])
    return loss, loss_reduced


def forward_step(data_iterator, model):
    args = get_args()

    batch = next(data_iterator)
    device = torch.device("cuda", args.local_rank)

    if multi_pattern_match(batch["dataset_tag"][0], ["t2i*"]):
        model_input_kwargs, bsz, seqlen = prepare_model_t2i_inputs(batch, device)
    elif multi_pattern_match(batch["dataset_tag"][0], ["lm*"]):
        model_input_kwargs, bsz, seqlen = prepare_model_lm_inputs(batch, device)
    elif multi_pattern_match(batch["dataset_tag"][0], ["mmu*"]):
        model_input_kwargs, bsz, seqlen = prepare_model_mmu_inputs(batch, device)
        # for k, v in model_input_kwargs.items():
        #     if "diffusion" in k:
        #         continue
        #     print(k, v.shape if isinstance(v, torch.Tensor) else v)
    else:
        raise NotImplementedError(f"dataset_tag `{batch['dtype']}` not recognized in forward_step.")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output = model(**model_input_kwargs)

    # accumulate samples per step
    # samples = batch["n_samples"].sum().item()
    # key = batch['dataset_tag'][0]

    return output.logits, partial(loss_func, output.losses)


def extra_args_provider(parser: argparse.ArgumentParser):
    from ..config import add_core_args

    parser = add_core_args(parser, ptm="v2")
    return parser


# =========================================================================
#       Entry point for training Gemini-MoE-A13B with AngelPTM v2
# =========================================================================
def launch(frozen_args):
    from megatron.core.enums import ModelType
    from megatron.training import inprocess_restart
    from megatron.training import pretrain

    # Optionally enable in-process restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        DatasetsProvider(),
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults=frozen_args,
        extra_args_provider=extra_args_provider,
        store=store,
    )
