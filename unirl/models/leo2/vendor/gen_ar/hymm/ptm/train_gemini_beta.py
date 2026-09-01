import gc
from functools import partial
import copy
import warnings

import torch
import torch.utils

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from hymm.config import *
from hymm.models import build_model
from hymm.ptm.training import pretrain
from hymm.utils.lr_schedules import add_tuning_arguments
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.utils.helpers import multi_pattern_match
from hymm.ptm.ptm_gemini_helper import get_gemini_trainer, initialize_gemini_trainer

from megatron import get_args, get_timers, print_rank_0, mpu
from megatron.utils import average_losses_across_data_parallel_group

gc.disable()
gc.set_threshold(7000, 100, 100)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="deepspeed")
warnings.filterwarnings("ignore", category=UserWarning, module="megatron")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformer_engine")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformer_engine")

@gc.callbacks.append
def callback(phase, info):
    if phase == 'stop':
        if info['collected']:
            #if torch.distributed.is_initialized():
            #    print(f"🟢 (PID={os.getpid()}) rank:{torch.distributed.get_rank()}, GC({info["generation"]}代) 已收集：{info['collected']}")
            #else:
            #    print(f"🟢 (PID={os.getpid()}) GC({info["generation"]}代) 已收集：{info['collected']}")
            pass
        if info["generation"] == 2:
            if torch.distributed.is_initialized():
                print(f"🔴   (PID={os.getpid()}), rank:{torch.distributed.get_rank()}, pprank:{mpu.get_pipeline_model_parallel_rank()}, 存活对象数目：{len(gc.get_objects())}")
            else:
                print(f"🔴   (PID={os.getpid()}), rank:{torch.distributed.get_rank()}, 存活对象数目：{len(gc.get_objects())}")

def model_provider(pre_process=True, post_process=True):
    """ Build the model. """

    args = get_args()
    # model_provider is called first in pretrain, so add parse_data_yaml in here
    parse_data_yaml(args) # update args form config-yaml for data
    args = sanity_check_args(args)

    dict_args = EasyDict(vars(copy.copy(args)))

    # for count samples per step
    args.consumed_samples_per_step = {}

    # gemini_trainer initialize dataloader, extra_model and data_iterator
    # align with base_trainer.py, Keep the order: dataloader -> model & optimizer -> extra model -> evaluator
    gemini_trainer = initialize_gemini_trainer(dict_args)
    # actually call MultiModalGeminiBetaTrainer.build_dataloader
    gemini_trainer.build_dataloader()
    dataset_tag = gemini_trainer.cur_key
    if multi_pattern_match(dataset_tag, ["t2i"]):
        assert "attn_type" in args and args.attn_type == 'flex'
        assert "t2i_batch_size" in args and args.t2i_batch_size is not None
    else:
        assert f"{dataset_tag}_attn_type" in args and getattr(args, f"{dataset_tag}_attn_type") == 'flex'
        assert f"{dataset_tag}_batch_size" in args and getattr(args, f"{dataset_tag}_batch_size") is not None

    if gemini_trainer.dataset_dict[dataset_tag].sequence_pack:
        args.micro_batch_size = 1
        args.seq_length = args.block_size

    elif multi_pattern_match(dataset_tag, ["t2i"]):
        args.micro_batch_size = args.t2i_batch_size
        args.seq_length = getattr(args, 'system_prompt_token_length', 0) + args.image_token_length + max(args.text_token_length, getattr(args, 'text_cot_token_length', 0))

    elif multi_pattern_match(dataset_tag, ["t2i*"]):
        args.micro_batch_size = getattr(args, f"{dataset_tag}_batch_size")
        args.seq_length = getattr(args, f"{dataset_tag}_system_prompt_token_length", 0) + getattr(args, f"{dataset_tag}_image_token_length") + max(
            getattr(args, f"{dataset_tag}_text_token_length"),
            getattr(args, f"{dataset_tag}_text_cot_token_length", 0)
        )

    elif multi_pattern_match(dataset_tag, ["interleave*", "pair*"]):
        assert f"{dataset_tag}_max_length" in args and getattr(args, f"{dataset_tag}_max_length") is not None
        args.micro_batch_size = getattr(args, f"{dataset_tag}_batch_size")
        args.seq_length = getattr(args, f"{dataset_tag}_max_length")

    elif multi_pattern_match(dataset_tag, ["lm*", "mmu*"]):
        assert f"{dataset_tag}_token_length" in args and getattr(args, f"{dataset_tag}_token_length") is not None
        args.micro_batch_size = getattr(args, f"{dataset_tag}_batch_size")
        args.seq_length = getattr(args, f"{dataset_tag}_token_length")

    else:
        raise ValueError(f"dataset_tag: {dataset_tag} not in [t2i*, mmu*, lm, interleave, pair*]")
    print(f"rank:{torch.distributed.get_rank()}, dprank:{mpu.get_data_parallel_rank()}, pprank:{mpu.get_pipeline_model_parallel_rank()}, data_type:{dataset_tag}, micro_batch_size:{args.micro_batch_size}, seq_length:{args.seq_length}")

    see_memory_usage(f"Before Building Model", force=True)
    assert args.deepspeed

    # TODO
    args.padded_vocab_size = 128256  #still need to keep the word embedding for ptm ckpt loading, but it can be deleted after loading
    args.consumed_image_epoch = 1
    args.consumed_video_epoch = 1
    args.consumed_text_epoch = 1
    args.consumed_image_samples_per_dp = 0
    args.consumed_video_samples_per_dp = 0
    args.consumed_text_samples_per_dp = 0
    args.consumed_image_samples_per_dp_total = 0
    args.consumed_video_samples_per_dp_total = 0

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 0,
                             mpu=mpu):
        factor_kwargs = {'device': torch.device("cuda", args.local_rank), 'dtype': PRECISION_TO_TYPE[args.precision]}
        print_rank_0(f"before build_model, args:{args}, dict_args:{dict_args}")
        model, model_settings = build_model(dict_args, **factor_kwargs)

    see_memory_usage(f"After Building Model", force=True)

    # move set_manual_seed to pretrain, after setup_model_and_optimizer. avoid the influence of load ckpt on random state
    # set_manual_seed(args.seed + mpu.get_data_parallel_rank())

    return model


def extra_models_provider(model):
    """ actually call build_extra_model in gemini trainer """

    # only first stage and last stage build vae, if no pipeline, first_stage/last_stage is True on all ranks
    if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
        skip_build_vae = False
    else:
        skip_build_vae = True

    gemini_trainer = get_gemini_trainer()
    # actually call GeminiTrainerAlphaMultiModal.build_extra_model
    see_memory_usage(f"Before build_extra_model", force=True)
    gemini_trainer.build_extra_model(skip_build_vae=skip_build_vae)
    see_memory_usage(f"After build_extra_model", force=True)
    return None


def broadcast_data(data_iterator):
    """
        maybe don't need to broadcast
    """
    if data_iterator is not None:
        ## ------------ get batch ------------
        batch = next(data_iterator)
    else:
        batch = {}

    return batch


def loss_func(loss_dict, moe_loss, output_tensor):
    args = get_args()
    if not mpu.is_pipeline_last_stage() and max(args.num_experts) > 1:
        # when moe + pp, return moe loss only
        assert args.mt_pipeline_parallel is True, "loss func pp+moe check failed"
        loss = moe_loss[0] * args.moe_loss_coeff
        loss_dict = {'moe loss': moe_loss[0]}
        if args.use_z_loss:
            loss += moe_loss[1] * args.z_loss_coeff
            loss_dict['z loss'] = moe_loss[1]
        loss_dict['exp capacity rate'] = moe_loss[2] / args.num_layers * args.expert_interval
        return loss, loss_dict

    loss = loss_dict['loss'].mean().float()
    if max(args.num_experts) > 1:
        loss_dict['moe loss'] = moe_loss[0]
        loss = loss + moe_loss[0] * args.moe_loss_coeff
        if args.use_z_loss:
            loss += moe_loss[1] * args.z_loss_coeff
        loss_dict['exp capacity rate'] = moe_loss[2] / args.num_layers * args.expert_interval

    # text_loss
    if 'text_loss' in loss_dict:
        text_loss = loss_dict['text_loss'].mean().float()
        text_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    else:
        text_loss = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        text_count = torch.tensor(0, dtype=loss.dtype, device=loss.device)
    # image_loss
    if 'image_loss' in loss_dict:
        image_loss = loss_dict['image_loss'].mean().float()
        image_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    else:
        image_loss = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        image_count = torch.tensor(0, dtype=loss.dtype, device=loss.device)
    # moe_loss
    if 'moe loss' in loss_dict:
        moe_loss = loss_dict['moe loss'].mean().float()
        exp_capacity_rate = loss_dict['exp capacity rate'].mean().float()
        moe_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    else:
        moe_loss = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        exp_capacity_rate = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        moe_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    
    # text_loss
    if 'text_loss' in loss_dict:
        text_loss = loss_dict['text_loss'].mean().float()
        text_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    else:
        text_loss = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        text_count = torch.tensor(0, dtype=loss.dtype, device=loss.device)
    
    # mmu_text_loss
    if 'mmu_text_loss' in loss_dict:
        mmu_text_loss = loss_dict['mmu_text_loss'].mean().float()
        mmu_text_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    else:
        mmu_text_loss = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        mmu_text_count = torch.tensor(0, dtype=loss.dtype, device=loss.device)
    
    # t2i_text_loss
    if 't2i_text_loss' in loss_dict:
        t2i_text_loss = loss_dict['t2i_text_loss'].mean().float()
        t2i_text_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    else:
        t2i_text_loss = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        t2i_text_count = torch.tensor(0, dtype=loss.dtype, device=loss.device)
    
    # lm_text_loss
    if 'lm_text_loss' in loss_dict:
        lm_text_loss = loss_dict['lm_text_loss'].mean().float()
        lm_text_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    else:
        lm_text_loss = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        lm_text_count = torch.tensor(0, dtype=loss.dtype, device=loss.device)

    averaged_losses = average_losses_across_data_parallel_group([loss, text_loss, image_loss,
                                                                 text_count, image_count,
                                                                 moe_count, moe_loss, exp_capacity_rate,
                                                                 mmu_text_loss, mmu_text_count, t2i_text_loss, t2i_text_count, lm_text_loss, lm_text_count])
    loss_dict['lm loss'] =  averaged_losses[0]
    loss_dict['text_loss'] = averaged_losses[1] / max(averaged_losses[3], 1)
    loss_dict['image_loss'] = averaged_losses[2] / max(averaged_losses[4], 1)
    loss_dict['mmu_text_loss'] = averaged_losses[8] / max(averaged_losses[9], 1)
    loss_dict['t2i_text_loss'] = averaged_losses[10] / max(averaged_losses[11], 1)
    loss_dict['lm_text_loss'] = averaged_losses[12] / max(averaged_losses[13], 1)
    if 'moe loss' in loss_dict:
        loss_dict['moe loss'] = averaged_losses[6] / max(averaged_losses[5], 1)
        loss_dict['exp capacity rate'] = averaged_losses[7] / max(averaged_losses[5], 1)
    return loss, loss_dict


def forward_step(data_iterator, model, extra_models, teacher_model=None, valid=False):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    # Get the batch.
    timers('batch-generator').start()

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        batch = next(data_iterator)
    else:
        batch = next(data_iterator[0])

    device = torch.device("cuda", args.local_rank)

    # only first stage and last stage run vae encode. if no pipeline, first_stage/last_stage is True on all ranks
    if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
        skip_vae_encode = False
    else:
        skip_vae_encode = True
    gemini_trainer = get_gemini_trainer()
    model_input_kwargs, cur_batch_size, n_tokens = gemini_trainer.prepare_model_inputs(batch, device, skip_vae_encode=skip_vae_encode)

    # torch.distributed.barrier()
    timers('batch-generator').stop()

    # Predict the noise residual
    timers('model-forward').start()

    with torch.autocast(device_type="cuda", dtype=PRECISION_TO_TYPE[args.precision], enabled=True):
        output_tensor, loss_dict, moe_loss = model(**model_input_kwargs)

    timers('model-forward').stop()

    # accumulate samples per step
    samples = batch["n_samples"].sum().item()
    key = batch['dtype'][0]
    args.consumed_samples_per_step.setdefault(key, 0)
    args.consumed_samples_per_step[key] += samples

    return output_tensor, partial(loss_func, loss_dict, moe_loss)


def train_valid_test_datasets_provider(train_val_test_num_samples, extra_models=None):
    """Build train, valid, and test datasets."""
    return True, None, None


def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset.
    """
    if dataset is None:
        # if valid or test, return None iter
        # return (None for _ in iter(int, 1))
        return None

    gemini_trainer = get_gemini_trainer()
    data_loader = gemini_trainer.get_data_iterator()

    # must add '[]', in order to adapt the code "if data_iter:" in deepspeed/runtime/pipe/engine.py
    return [data_loader]
    # return data_loader


def add_dit_args(parser: argparse.ArgumentParser):
    kwargs = dict(ptm=True)
    # parser = add_logging_args(parser, **kwargs)
    parser = add_model_args(parser, **kwargs)
    parser = add_extra_models_args(parser, **kwargs)
    parser = add_denoise_schedule_args(parser, **kwargs)
    parser = add_data_args(parser, **kwargs)
    parser = add_data_yaml_args(parser, **kwargs)
    parser = add_deepspeed_args(parser, **kwargs)
    parser = add_ema_args(parser, **kwargs)
    parser = add_training_args(parser, **kwargs)
    parser = add_evaluation_args(parser, **kwargs)
    parser = add_tools_args(parser, **kwargs)
    parser = add_tuning_arguments(parser, **kwargs)

    group = parser.add_argument_group(title="dit")
    group.add_argument("--build_pretraining_data_loader_func", default=build_pretraining_data_loader)
    # 开启 log_validation, 不开启的话不会在训练中validation, ${EVAL_INTERVAL}也不会生效
    group.add_argument("--dit-valid", action="store_true", default=False, help="whether validation in training")
    # 关闭 autocast
    group.add_argument("--disable-torch-amp", action="store_true", help="disable torch amp.")
    # 使用 fp32 算子
    group.add_argument("--force-fp32-ops", action='store_true')
    # 使用 TE 算子, 开启 TP / SP
    group.add_argument("--ptm-v2", action="store_true", help="enable ptm v2.")
    # sequence-parallel padding
    group.add_argument("--sequence-parallel-padding", action="store_true", help="padding img for sequence parallel")
    group.add_argument("--sp-img-pad-size", type=int, default=-1,
                       help="img padded size for sequence parallel, don't need setting.")
    group.add_argument("--sp-txt-pad-size", type=int, default=-1,
                       help="txt padded size for sequence parallel, don't need setting.")
    group.add_argument("--sp-x-pad-size", type=int, default=-1,
                       help="x padded size for sequence parallel, don't need setting.")
    # 评估流程使用，只跑validation不训练, 优先级最高的参数, 谨慎开启
    group.add_argument("--only-validation", action="store_true", default=False,
                       help="validation before train, and exit after validation.")
    # 评估流程使用，只跑视频的sample validation不训练
    group.add_argument("--offline-sample-video", action="store_true", default=False,
                       help="validation before train, and exit after validation.")
    group.add_argument('--consumed-video', type=int, default=0, help='consumed video')
    group.add_argument('--consumed-img1', type=int, default=0, help='consumed img1')
    group.add_argument('--consumed-img2', type=int, default=0, help='consumed img2')
    group.add_argument('--consumed-img3', type=int, default=0, help='consumed img3')
    group.add_argument('--dummy-number', type=int, default=0, help='dummy number')
    group.add_argument("--vae-encode-chunk-size", type=int, default=-1, help="vae encode chunk size.")
    # choose some ops to not be recomputed
    group.add_argument('--selective-checkpoint', action='store_true', help='choose some ops to not be recomputed')
    ## a list of op names which not be recomputed
    # parser.add_argument('--skip-recompute-ops', nargs='+', type=str, default=['op_fused_attn_fwd.default'], help='a list of op names which not be recomputed')
    # layers enable selective-checkpoint
    group.add_argument('--selective-checkpoint-layers-range', type=str,
                       help='layers enable selective-checkpoint, for example: 3-5 means [3, 5)')
    # force enable qkv_weight_interleaved when ptm_v2 and tp_size = 1
    group.add_argument('--force-enable-qkv-interleaved', action='store_true',
                       help='force enable qkv_weight_interleaved when ptm_v2 and tp_size = 1')

    # align yaml
    group.add_argument("--block_size", type=int,
                    help="align model_kwargs.block_size")

    group.add_argument("--use-ptm", action="store_true", help="use ptm in model")
    group.add_argument("--use-final-time-embed", action="store_true", help="use final time embed")
    group.add_argument("--sync-ss-after-load", action="store_true",
                    help="Sync scalar states after load ckpt. It fixes the sync problem of ss when training "
                        "multiple epochs.")
    return parser


def train():
    pretrain(train_valid_test_dataset_provider=train_valid_test_datasets_provider,
             model_provider=model_provider,
             forward_step_func=forward_step,
             extra_args_provider=add_dit_args,
             extra_models_provider=extra_models_provider,
             )


if __name__ == "__main__":
    train()
