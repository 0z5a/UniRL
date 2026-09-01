# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time
import json
import psutil
import os
import copy
from megatron.utils import lazy_gc, average_losses_across_data_parallel_group, get_all_moe_loss
import torch.distributed

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_current_micro_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank, is_first_rank
from megatron import update_num_microbatches
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron import print_rank
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.model import ModelType
from megatron.model import GPTModel
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.learning_rates import AnnealingLR
from megatron.optimizer.adafactor import AdafactorLRSchedule
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model.distributed_powersgd import DistributedDataParallelWithPowerSGD as LocalDDPWithPowerSGD
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.data.gpt_dataset import check_stream_data_prefix, remove_consume_stream_data
from megatron.utils import calc_params_l2_norm, convert_lora_to_linear_layer, only_optimize_lora_parameters, unfuse_lora_from_linear_layer
from megatron.schedules import forward_backward_no_pipelining
from megatron.schedules import forward_backward_pipelining_without_interleaving
from megatron.schedules import forward_backward_zero_bubble_pipelining_without_interleaving
from megatron.schedules import forward_backward_pipelining_with_interleaving
from megatron.fab_schedules import fab_forward_backward_no_pipelining, fab_forward_backward_pipelining_with_interleaving, fab_forward_backward_pipelining_without_interleaving
from megatron.mmp_schedules import forward_backward_no_pipelining_with_mmp_encoder_parallel, forward_backward_pipelining_with_interleaving_with_mmp_encoder_parallel, forward_backward_pipelining_without_interleaving_with_mmp_encoder_parallel
from megatron.vlm_schedules import forward_backward_no_pipelining_with_vlm_independent_parallelism, forward_backward_pipelining_with_interleaving_with_vlm_independent_parallelism, forward_backward_pipelining_without_interleaving_with_vlm_independent_parallelism
from megatron.utils import report_memory, flops_calculator, set_model_weight_by_user
from megatron.user_optim import UserOptim
from megatron.global_vars import set_validate_batches_states, set_train_precision, get_train_precision, reset_args
from megatron.activation_offloading import get_ptm_offloading_context
from megatron.maintain import calc_mfu
from megatron.vlm_utils import setup_vit_model_and_optimizer, save_vit_checkpoint, vlm_optimizer_step, set_vit_model_state
from megatron.arguments import process_moe_args
from megatron.custom_layers.utils import clear_te_cache
import deepspeed

import hymm.utils.lr_schedules as lr_schedules
from hymm.ptm.ptm_gemini_helper import get_gemini_trainer
from hymm.utils.torch_utils import set_manual_seed

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))



def init_pipeline_recorder(args):
    if args.profile_pipeline_all_ranks or (mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0):
        # init log file
        from megatron.pipeline_recorder import get_pp_recorder_log_file
        log_file = get_pp_recorder_log_file(args, mpu)
        torch.cuda.synchronize()
        with open(log_file, "w") as f:
            f.write(f"S {time.time()}\n")
    return


def _call_callback_hooks(model, hook_name: str, iteration: int, *args, **kwargs) -> None:
    if not hasattr(model, "callbacks"):
        return

    if hasattr(model, "global_step"):
        model.global_step = iteration

    fn = getattr(model, hook_name)
    if callable(fn):
        print_rank_0(f"{model.__class__.__name__}: calling with hook: {hook_name}")
        batch = model.current_batch if hasattr(model, "current_batch") else None
        batch_idx = model.batch_idx if hasattr(model, "batch_idx") else None
        trainer = model.trainer if hasattr(model, "trainer") else None
        fn(*args, **kwargs, batch=batch, batch_idx=batch_idx)

    for callback in model.callbacks:
        fn = getattr(callback, hook_name)
        if callable(fn):
            print_rank_0(f"{model.__class__.__name__}: calling callback {callback.__class__.__name__} with hook: {hook_name}")
            batch = model.current_batch if hasattr(model, "current_batch") else None
            batch_idx = model.batch_idx if hasattr(model, "batch_idx") else None
            trainer = model.trainer if hasattr(model, "trainer") else None
            fn(trainer, model, *args, **kwargs, batch=batch, batch_idx=batch_idx)


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             forward_step_func,
             model_type = ModelType.encoder_or_decoder,
             extra_args_provider=None,
             args_defaults={},
             extra_models_provider=None,
             multimodal_encoder_model_provider=None,
             encoder_llm_comm_tensor_shape_func=None, **kwargs):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    if args.profile_pipeline:
        init_pipeline_recorder(args)

    global _DATASET_CLASS
    _DATASET_CLASS = kwargs.get('dataset_class', None)

    args.is_zero_stage_3 = False

    if args.deepspeed:
        args.deepspeed_configuration = json.load(
            open(args.deepspeed_config, 'r', encoding='utf-8'))
        if "curriculum_learning" in args.deepspeed_configuration and \
            "enabled" in args.deepspeed_configuration["curriculum_learning"]:
            args.curriculum_learning = args.deepspeed_configuration[ \
                "curriculum_learning"]["enabled"]
        if args.curriculum_learning and not args.no_pipeline_parallel:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                import CurriculumScheduler
            args.curriculum_scheduler = CurriculumScheduler( \
                args.deepspeed_configuration["curriculum_learning"])

        if "zero_optimization" in args.deepspeed_configuration and \
            "stage" in args.deepspeed_configuration["zero_optimization"]:
            if args.deepspeed_configuration["zero_optimization"]["stage"] == 3:
                args.is_zero_stage_3 = True

    if args.use_multi_validation_dataset != None:
        dataset_name_and_path = []
        for i in range(len(args.use_multi_validation_dataset)):
            dataset_info = args.use_multi_validation_dataset[i].split(":")
            if len(dataset_info) == 1:
                dataset_fake_name_and_path = f'temp_valid_dataset_name_{i}:{args.use_multi_validation_dataset[i]}'
                dataset_name_and_path.append(dataset_fake_name_and_path)
            else:
                dataset_name_and_path.append(args.use_multi_validation_dataset[i])
        args.use_multi_validation_dataset = dataset_name_and_path

    if args.is_zero_stage_3:
        # Data stuff.
        timers('train/valid/test-data-iterators-setup', log_level=0).start()
        # must set args.iteration as None, otherwise iteration will not be loaded from ckpt
        args.iteration = None
        # load checkpoint file to get real iteration
        if args.load is not None:
            #TODO: check checkpoint exists
            latest_path = os.path.join(args.load, "latest")
            if os.path.exists(latest_path) and os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    tag = fd.read().strip().replace("global_step", "")
                    args.iteration = int(tag)
                    print_rank_0(f"iteration={args.iteration}")
            extra_path = os.path.join(args.load, "extra_configs")
            if os.path.exists(extra_path) and os.path.isfile(extra_path):
                with open(extra_path, "r") as fd:
                    extra_lines = fd.readlines()
                    extra_dict = {}
                    for l in extra_lines:
                        key, value = l.strip('\n').split(':')
                        extra_dict[key] = int(value)
                    if 'consumed_train_samples' in extra_dict:
                        args.consumed_train_samples = extra_dict['consumed_train_samples']
                        print_rank_0(f"consumed_train_samples={args.consumed_train_samples}")
                    if 'consumed_val_samples' in extra_dict:
                        args.consumed_val_samples = extra_dict['consumed_val_samples']
                        print_rank_0(f"consumed_val_samples={args.consumed_val_samples}")
                    else:
                        args.consumed_val_samples = (args.iteration // args.eval_interval) * \
                             args.eval_iters * args.valid_global_batch_size
                        print_rank_0(f"consumed_val_samples={args.consumed_val_samples}")

        # Model, optimizer, and learning rate.
        timers('model-and-optimizer-setup', log_level=0).start()
        model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider, model_type, teacher=False, multimodal_encoder_model_provider_func=multimodal_encoder_model_provider)
        vit_model, vit_optimizer, vit_lr_scheduler = setup_vit_model_and_optimizer(multimodal_encoder_model_provider, get_learning_rate_scheduler)
        timers('model-and-optimizer-setup').stop()
        print_datetime('after model, optimizer, and learning rate '
                       'scheduler are built')

        if extra_models_provider is not None:
            extra_models = extra_models_provider(model)

        # After build_extra_model, we set different seed for each process.
        set_manual_seed(args.seed + mpu.get_data_parallel_rank())

        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator = []
            valid_data_iterator = []
            test_data_iterator = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                iterators = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider, extra_models)
                train_data_iterator.append(iterators[0])
                valid_data_iterator.append(iterators[1])
                test_data_iterator.append(iterators[2])
        else:
            train_data_iterator, valid_data_iterator, test_data_iterator \
                = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider, extra_models)
        timers('train/valid/test-data-iterators-setup').stop()
        print_datetime('after dataloaders are built')
    else:
        # Model, optimizer, and learning rate.
        timers('model-and-optimizer-setup', log_level=0).start()
        model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider, model_type, teacher=False)
        vit_model, vit_optimizer, vit_lr_scheduler = setup_vit_model_and_optimizer(multimodal_encoder_model_provider, get_learning_rate_scheduler)
        timers('model-and-optimizer-setup').stop()
        print_datetime('after model, optimizer, and learning rate '
                       'scheduler are built')
        if extra_models_provider is not None:
            extra_models = extra_models_provider(model)

        # After build_extra_model, we set different seed for each process.
        set_manual_seed(args.seed + mpu.get_data_parallel_rank())

        # Data stuff.
        timers('train/valid/test-data-iterators-setup', log_level=0).start()
        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator = []
            valid_data_iterator = []
            test_data_iterator = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                iterators = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider, extra_models)
                train_data_iterator.append(iterators[0])
                valid_data_iterator.append(iterators[1])
                test_data_iterator.append(iterators[2])
        else:
            train_data_iterator, valid_data_iterator, test_data_iterator \
                = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider, extra_models)
        timers('train/valid/test-data-iterators-setup').stop()
        print_datetime('after dataloaders are built')

    # for use_multiple_validation_dataset
    if valid_data_iterator is not None and args.use_multi_validation_dataset and \
      args.virtual_pipeline_model_parallel_size:
        valid_data_iterator = [list(t) for t in zip(*valid_data_iterator)]

    set_model_weight_by_user(args.directory_of_model_weight_set_files, model)

    #save the checkpoint when load it! for converting ckpts
    if args.save_after_load or args.convert_ckpt:
        print_datetime('Saving the model after loads it!')
        # save trainer_ptm_wrapper.ss TODO: implement the serialization function of ss, instead of calling to_dict before save_checkpoint. by youngfyang
        gemini_trainer = get_gemini_trainer()
        args.scalar_states = gemini_trainer.ss.to_dict()
        save_checkpoint(args.iteration+1, model, optimizer, lr_scheduler, save_tensors_mode=args.save_tensors_mode)
        save_vit_checkpoint(args.iteration+1, args.save_tensors_mode)
        print_datetime('models! saved!')
        if args.convert_ckpt:
            sys.exit()


    teacher_model = None
    if args.mos: # Set up teacher model
        teacher_model = setup_teacher_model(args, model_provider)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'])
    print_rank_0('training ...')
    for_stream_ds=None
    if args.use_stream_dataset or args.use_iteration_dataset:
        for_stream_ds = [train_data_iterator, valid_data_iterator, test_data_iterator]

    iteration = 0
    # After model_provider executed, we can get the initialized gemini_trainer
    gemini_trainer = get_gemini_trainer()
    gemini_trainer.ss.current_run_update_steps = 0

    eval_forward_step_func = kwargs.get('eval_forward_step_func', None)

    if args.use_multi_validation_dataset != None and eval_forward_step_func is None:
        eval_forward_step_func=[]
        for i in range(len(args.use_multi_validation_dataset)):
            dataset_info = args.use_multi_validation_dataset[i].split(":")
            assert len(dataset_info) == 2 , "every dataset_dir should get a name to tag it, so the input of --use_multi_validation_dataset should be like dataset_num:dataset_dir"
            if dataset_info[0] in ["mmlu","ceval"]:
                from megatron.data.accuracy_dataset import acc_eval_forward_step
                eval_forward_step_func.append(acc_eval_forward_step)
            else:
                eval_forward_step_func.append(forward_step_func)
    elif args.use_multi_validation_dataset==None:
        eval_forward_step_func = forward_step_func

    if args.valid_first:
        prefix = 'iteration {}'.format(iteration)
        evaluate_and_print_results(prefix, eval_forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, False, encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)

    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator,
                          for_stream_ds = for_stream_ds,
                          custom_inner_while_train_func = kwargs.get('custom_inner_while_train_func', None),
                          teacher_model=teacher_model, extra_models=extra_models, eval_forward_step_func=eval_forward_step_func,
                          encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)
    print_datetime('after training is done')

    if args.do_test and not args.use_stream_dataset:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, eval_forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, False, teacher_model=teacher_model,
                                   encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)

    if args.save and iteration != 0 and args.eval_save_model_key == None:
        if args.use_lora:
            convert_lora_to_linear_layer(model[0].module)
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args.ckpt_nums, save_tensors_mode=args.save_tensors_mode)
        save_vit_checkpoint(iteration, args.save_tensors_mode)

    if args.do_test and not args.use_stream_dataset:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, eval_forward_step_func,
                                   test_data_iterator, model,
                                   0, True, teacher_model=teacher_model,
                                   encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)

def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size
        args.num_micro_batches = get_num_microbatches()
        args.current_global_batch_size = get_current_global_batch_size()

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        args.num_micro_batches = get_num_microbatches()
        args.current_global_batch_size = get_current_global_batch_size()
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))

def setup_torch_ddp(
    model_module,
    device_ids,
    output_device,
    process_group,
):
    import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as PowerSGD

    args = get_args()

    ddp_model = torchDDP(
        model_module,
        device_ids=device_ids,
        output_device=output_device,
        process_group=process_group,
    )

    if args.enable_powersgd:
        state = PowerSGD.PowerSGDState(
            process_group=process_group,
            matrix_approximation_rank=args.powersgd_matrix_approximation_rank,
            start_powerSGD_iter=args.powersgd_start_iter,
            warm_start=args.powersgd_warm_start,
            use_error_feedback=args.powersgd_error_feedback,
            batch_tensors_with_same_shape=args.batch_tensors_powersgd,
        )
        ddp_model.register_comm_hook(state, PowerSGD.powerSGD_hook)

    return ddp_model


def setup_local_ddp(
    model_module,
    accumulate_allreduce_grads_in_fp32,
    use_contiguous_buffers,
):
    args = get_args()
    if args.enable_powersgd:
        ddp_model = LocalDDPWithPowerSGD(
            model_module,
            accumulate_allreduce_grads_in_fp32,
            use_contiguous_buffers,
            args.powersgd_matrix_approximation_rank,
            args.powersgd_start_iter,
            args.powersgd_warm_start,
            args.powersgd_error_feedback)
    else:
        ddp_model = LocalDDP(
            model_module,
            accumulate_allreduce_grads_in_fp32,
            use_contiguous_buffers)
    return ddp_model


def setup_teacher_model(args, model_provider):
    assert not args.mmp_encoder_parallel, "Dont support distill with mmp parallel"
    print_rank_0('***>>>>> Student model checkpoint iteration:{}'.format(args.iteration))
    origin_args = copy.deepcopy(args)

    print_rank_0('***>>>>> Setting up the teacher model')

    mpu.reset_global_vals_of_num_layers()
    for key, value in vars(origin_args).items():
        if 'teacher' in key and value is not None:
            args.__dict__[key[: -(len('teacher') + 1)]] = value

    if max(args.num_experts) > 1 and (not mpu.expert_parallel_is_initialized() or not mpu.expert_and_tensor_parallel_is_initialized()):
        args = process_moe_args(args)
        mpu.initialize_expert_parallel(args.tensor_model_parallel_size,
                                       args.pipeline_model_parallel_size,
                                       args.fp8_e4m3 or args.fp8_hybrid,
                                       args.context_parallel_size,
                                       args)
        if args.seed is not None and args.seed > 0:
            # Ensure that different pipeline MP stages get different seeds.
            seed = args.seed + (100 * mpu.get_pipeline_model_parallel_rank())
            if torch.cuda.device_count() > 0:
                mpu.expert_parallel_cuda_manual_seed(seed)

    teacher_model, _, _ = load_model_weights_only(model_provider)
    print_rank_0('***>>>>> Teacher model:{}'.format(teacher_model))

    reset_args(origin_args)
    mpu.reset_global_vals_of_num_layers()

    return teacher_model


def get_model(model_provider_func, model_type = ModelType.encoder_or_decoder, wrap_with_ddp=True, multimodal_encoder_model_provider_func=None):
    """Build the model."""
    args = get_args()
    if mpu.is_mmp_encoder_stage():
        model = multimodal_encoder_model_provider_func()
        if not isinstance(model, list):
            model = [model]
        return model
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:

        assert not args.mmp_encoder_parallel, f"multimodal paralell is not tested on VPP currently"
        model = []
        if model_type == ModelType.encoder_and_decoder:
            pre_process = False
            post_process = False
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                # Set pre_process and post_process only after virtual rank is set.
                rank = mpu.get_pipeline_model_parallel_rank()
                pre_process = mpu.is_pipeline_first_stage() or mpu.is_pipeline_stage_at_split() # chunk[0] and chunk[split_chunk_idx] of rank 0. The first encoder layer and the first decoder layer.
                post_process = mpu.is_pipeline_last_stage_before_split() or mpu.is_pipeline_last_stage() # chunk[split_chunk_idx-1] and chunk[-1] of rank -1. The last encoder layer and the last decoder layer.
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()

                this_model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    add_encoder=add_encoder,
                    add_decoder=add_decoder)
                this_model.model_type = model_type
                model.append(this_model)
        else:
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                this_model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process
                )
                this_model.model_type = model_type
                model.append(this_model)
            # in order to Compatible deepspeed, convert model to torch.nn.ModuleList
            # TODO: be carefully, model name has changed
            model = torch.nn.ModuleList(model)
            model.model_type = model_type
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            assert not args.mmp_encoder_parallel, f"multimodal paralell is not tested for encoder_and_decoder currently"
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            if args.vit_type is not None:
                if args.vit_pipeline_model_parallel_split_rank is not None:
                    assert mpu.get_pipeline_model_parallel_world_size() > 1, "pp_size must > 0 for vit-pp"
                    rank = mpu.get_pipeline_model_parallel_rank()
                    world_size = mpu.get_pipeline_model_parallel_world_size()
                    if args.split_rank_llm_num_layers > 0:
                        gpt_trans_start_rank = args.vit_pipeline_model_parallel_split_rank
                    else:
                        gpt_trans_start_rank = args.vit_pipeline_model_parallel_split_rank + 1
                    vit_end_rank = args.vit_pipeline_model_parallel_split_rank
                    vit_pre_process = rank == 0
                    vit_post_process = rank == vit_end_rank
                    gpt_pre_process = vit_post_process
                    gpt_post_process = rank == (world_size - 1)
                    vit_process = rank <= vit_end_rank
                    gpt_process = rank >= gpt_trans_start_rank  # has transformer layer
                    args.vit_process = vit_process
                    model = model_provider_func(
                        vit_pre_process=vit_pre_process,
                        vit_post_process=vit_post_process,
                        gpt_pre_process=gpt_pre_process,
                        gpt_post_process=gpt_post_process,
                        vit_process=vit_process,
                        gpt_process=gpt_process)
                else:
                    vit_pre_process = pre_process
                    vit_post_process = pre_process
                    gpt_pre_process = pre_process
                    gpt_post_process = post_process
                    vit_process = pre_process
                    gpt_process = True
                    args.vit_process = vit_process
                    if args.vlm_independent_parallelism:
                        vit_process = False
                        vit_ppre_process = False
                        vit_post_process = False
                    model = model_provider_func(
                        vit_pre_process=vit_pre_process,
                        vit_post_process=vit_post_process,
                        gpt_pre_process=gpt_pre_process,
                        gpt_post_process=gpt_post_process,
                        vit_process=vit_process,
                        gpt_process=gpt_process)
            else:
                model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process
                )
        model.model_type = model_type

    if args.activation_offloading:
        def set_model_offloading_context(model, offloading_layers):
            offloading_ctx, offloading_commit_fn = get_ptm_offloading_context(
                offloading_layers,
                args.offloading_op_types,
                profiling=args.activation_profiling,
                activation_recompute=args.activation_recompute,
                offloading_minimum_tensor_size=args.offloading_minimum_tensor_size)
            model.offloading_ctx = offloading_ctx
            model.offloading_commit_fn = offloading_commit_fn

        if isinstance(model, (list, torch.nn.ModuleList)):
            for idx, model_chunk in enumerate(model):
                if not args.activation_profiling:
                    pp_rank = mpu.get_pipeline_model_parallel_rank()
                    offloading_layers = args.offloading_layers[pp_rank][idx]
                else:
                    offloading_layers = None
                set_model_offloading_context(model_chunk, offloading_layers)
        else:
            if not args.activation_profiling:
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    pp_rank = mpu.get_pipeline_model_parallel_rank()
                    offloading_layers = args.offloading_layers[pp_rank]
                else:
                    offloading_layers = args.offloading_layers
            else:
                offloading_layers = None
            set_model_offloading_context(model, offloading_layers)


    if not isinstance(model, list):
        model = [model]

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    if isinstance(model[0], torch.nn.ModuleList):
        args.allow_transformer_engine = all([type(m) == GPTModel for m in model[0]])
    else:
        from hymm.models.autoregressive.multimodal_transfusion import MultiModalTransfusion
        args.allow_transformer_engine = all([type(m) == GPTModel or MultiModalTransfusion for m in model])
    assert args.allow_transformer_engine or args.transformer_impl == 'local', \
        'Transformer Engine is only approved for GPT models'

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        if args.only_optimize_lora:
            only_optimize_lora_parameters(model_module, bool(args.extra_vocab_size))
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.ds_numel if hasattr(p,'ds_id') else p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    if args.deepspeed:
        return model

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = [setup_torch_ddp(model_module, device_ids=[i], output_device=i,
                                 process_group=mpu.get_data_parallel_group())
                 for model_module in model]
        return model

    if args.DDP_impl == 'local':
        model = [setup_local_ddp(model_module,
                                 args.accumulate_allreduce_grads_in_fp32,
                                 args.use_contiguous_buffers_in_ddp)
                 for model_module in model]
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        args.num_micro_batches = get_num_microbatches()
        args.current_global_batch_size = get_current_global_batch_size()
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            f'either train-iters or train-samples should be provided. but get {args.train_iters} and {args.train_samples}')

    if args.optimizer == 'adafactor':
        # adafactor don't need a lr_scheduler.
        lr_scheduler = AdafactorLRSchedule(optimizer, initial_lr=args.lr)
    else:
        if hasattr(args, 'lr_schedule') and args.lr_schedule == 'WarmupCosineLR':
            lr_scheduler = lr_schedules.WarmupCosineLR(
                optimizer,
                total_num_steps=args.train_iters,
                warmup_min_ratio=args.warmup_min_ratio,
                warmup_num_steps=args.warmup_num_steps,
                cos_min_ratio=args.cos_min_ratio)
        else:
            lr_scheduler = AnnealingLR(
                optimizer,
                max_lr=args.lr,
                min_lr=args.min_lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                decay_style=args.lr_decay_style,
                use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
                override_lr_scheduler=args.override_lr_scheduler,
                vit_max_lr=args.vit_lr,
                vit_min_lr=args.vit_min_lr)

    return lr_scheduler


def get_optimizer_and_scheduler(model, init_scheduler=False):
    optimizer = get_megatron_optimizer(model)
    scheduler = None
    if init_scheduler:
        scheduler = get_learning_rate_scheduler(optimizer)
    return optimizer, scheduler


def load_model_weights_only(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()
    print_rank_0('***>>>>> Args:{}'.format(args))

    model = get_model(model_provider_func)

    optimizer = None
    lr_scheduler = None

    if args.deepspeed:
        unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))
        params = get_megatron_optimizer(unwrapped_model)
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0],
            args = args,
            model_parameters=params,
            mpu=mpu,
        )

        model = [model]

    print_datetime('before load checkpoint')
    if args.load is not None:
        iteration = load_checkpoint(model, optimizer, lr_scheduler, strict=True, load_only_weights=True)

    print_datetime('after load checkpoint weights')

    return model, optimizer, lr_scheduler


def setup_model_and_optimizer(model_provider_func, model_type = ModelType.encoder_or_decoder, teacher=False, multimodal_encoder_model_provider_func=None):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type, multimodal_encoder_model_provider_func=multimodal_encoder_model_provider_func)

    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))

    if args.inference:
        #optimizer = None
        optimizer = get_megatron_optimizer(unwrapped_model)
        lr_scheduler = None
        args.train_iters = 1
    else:
        if teacher:
            optimizer = None
        else:
            optimizer, lr_scheduler = get_optimizer_and_scheduler(unwrapped_model, not args.deepspeed_optimizer)
    if args.rampup_batch_size is not None:
        update_train_iters(args)
    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        print_rank('initialize stage start to init deepspeed')
        pp = mpu.get_pipeline_model_parallel_world_size()
        if not args.no_pipeline_parallel and args.mt_pipeline_parallel:
            mt_pipeline_parallel = True
        elif args.mmp_encoder_parallel:
            mt_pipeline_parallel = True
        else:
            mt_pipeline_parallel = False

        if args.deepspeed_optimizer:
            print_rank_0("The optimizer is initialized from DeepSpeed, so does the lr scheduler.")
            params = get_megatron_optimizer(unwrapped_model)
            if args.user_optimizer:
                optimizer = UserOptim(unwrapped_model, args)
                model, optimizer, _, lr_scheduler = deepspeed.initialize(
                    model=model[0],
                    optimizer=optimizer,
                    args=args,
                    mpu=None if not mt_pipeline_parallel else mpu
                )
            else:
                if hasattr(args, 'no_init_optim'):
                    model, optimizer, _, lr_scheduler = deepspeed.initialize(
                        model=model[0],
                        infer_model_parameters=params if args.no_init_optim else None,
                        model_parameters=params if not args.no_init_optim else None,
                        args=args,
                        lr_scheduler=get_learning_rate_scheduler,
                        mpu=mpu if args.no_pipeline_parallel or mt_pipeline_parallel else None
                    )
                else:
                    model, optimizer, _, lr_scheduler = deepspeed.initialize(
                        model=model[0],
                        model_parameters=params,
                        args=args,
                        lr_scheduler=get_learning_rate_scheduler,
                        mpu=mpu if args.no_pipeline_parallel or mt_pipeline_parallel else None
                    )
        else:
            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=lr_scheduler,
                mpu=mpu if args.no_pipeline_parallel or mt_pipeline_parallel else None
            )
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            if not mt_pipeline_parallel:
                model.set_batch_fn(model.module._megatron_batch_fn)

            # mixed training is the same reason
            if args.mmp_encoder_parallel or args.use_mixed_training:
                pass # do not check grid for mmp, because we have a strict check in mpu
            else:
                assert model.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
                assert model.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
                assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank(with_context_parallel=True)
        model = [model]
        print_rank('initialize stage finish init deepspeed')

    args.init_iteration = 0
    if args.load is not None:
        timers = get_timers()
        # Extra barrier is added to make sure all ranks report the
        # max time.
        print_rank('sync before load checkpoint')
        torch.distributed.barrier()
        print_rank('ready to load checkpoint')
        timers('load-checkpoint', log_level=0).start()
        if args.mos:
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler, strict=False, load_only_weights=False)
        else:
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
        args.init_iteration = args.iteration
        torch.distributed.barrier()
        timers('load-checkpoint').stop()
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # Synchronize multimodal dataloader:
    #   For dataset A, epoch_consumed_samples are not synchronized between A-dataset ranks and Non-A-dataset ranks.
    #   A-dataset ranks will reset epoch_consumed_samples to zero after finishing an epoch,
    #   while Non-A-dataset ranks do not reset epoch_consumed_samples and keep increasing it.
    if hasattr(args, 'resume_dataloader') and args.resume_dataloader and args.sync_ss_after_load:
        gemini_trainer = get_gemini_trainer()
        all_gather_consumed_samples = [None for _ in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(all_gather_consumed_samples, gemini_trainer.ss.to_dict(),
                                            group=mpu.get_data_parallel_group())
        for key in all_gather_consumed_samples[0]["epoch_consumed_samples"].keys():
            epoch_consumed_samples = [
                ss_dict["epoch_consumed_samples"][key]
                for ss_dict in all_gather_consumed_samples
                if key in ss_dict["epoch_consumed_samples"]
            ]
            if len(epoch_consumed_samples) > 0:
                min_epoch_consumed_samples = min(epoch_consumed_samples)
                gemini_trainer.ss.epoch_consumed_samples[key] = min_epoch_consumed_samples

            consumed_epoch = [
                ss_dict["consumed_epoch"][key]
                for ss_dict in all_gather_consumed_samples
                if key in ss_dict["consumed_epoch"]
            ]
            if len(consumed_epoch) > 0:
                max_consumed_epoch = max(consumed_epoch)
                gemini_trainer.ss.consumed_epoch[key] = max_consumed_epoch

    # Print updated args for supplementary information.
    from megatron.arguments import _print_args
    _print_args(args)

    return model, optimizer, lr_scheduler


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler, teacher_model=None, extra_models=None,
               encoder_llm_comm_tensor_shape_func=None):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    if args.activation_offloading:
        forward_step_ori = forward_step_func
        def forward_step_offloading_wrapper(*args, **kwargs):
            model = args[1]
            if isinstance(model, deepspeed.DeepSpeedEngine):
                offloading_ctx = model.module.offloading_ctx
                offloading_commit_fn = model.module.offloading_commit_fn
            else:
                offloading_ctx =  model.offloading_ctx
                offloading_commit_fn = model.offloading_commit_fn

            with offloading_ctx:
                outputs = forward_step_ori(*args, **kwargs)
            if isinstance(outputs, (list, tuple)):
                outputs = list(outputs)
                outputs[0] = offloading_commit_fn(outputs[0])
            elif torch.is_tensor(outputs):
                outputs = offloading_commit_fn(outputs)
            else:
                raise ValueError("invalid forward step output, only support list/tuple or single tensor")
            return outputs

        forward_step_func = forward_step_offloading_wrapper

    # add flags for zerocache with grad acc
    args.is_last_micro_batch = False
    if args.deepspeed and args.ds_pipeline_enabled:
        skipped_iter = 0
        num_zeros_in_grad = 0
        # add flags for async grad copy in zero2 and zero3
        args.is_last_pp_micro_batch = False
        # deepspeed using megatron pipeline
        if args.mt_pipeline_parallel:
            assert isinstance(model[0], deepspeed.PipelineEngine)
            assert args.mmp_encoder_parallel or mpu.get_pipeline_model_parallel_world_size() > 1
            if args.virtual_pipeline_model_parallel_size is not None:
                if args.mmp_encoder_parallel:
                    forward_backward_func = forward_backward_pipelining_with_interleaving_with_mmp_encoder_parallel
                elif args.enable_forward_and_backward:
                    forward_backward_func = fab_forward_backward_pipelining_with_interleaving
                else:
                    forward_backward_func = forward_backward_pipelining_with_interleaving
                    if args.vlm_independent_parallelism:
                        forward_backward_func = forward_backward_pipelining_with_interleaving_with_vlm_independent_parallelism
                    assert get_num_microbatches() % args.pipeline_model_parallel_size == 0, \
                        'number of microbatches is not divisible by pipeline-parallel ' \
                        'size when using interleaved schedule'
            else:
                if args.mmp_encoder_parallel:
                    # TODO. merge to one fwd_bed_func
                    if args.pipeline_model_parallel_size > 1:
                        forward_backward_func = forward_backward_pipelining_without_interleaving_with_mmp_encoder_parallel
                    else:
                        forward_backward_func = forward_backward_no_pipelining_with_mmp_encoder_parallel
                elif args.vlm_independent_parallelism:
                    forward_backward_func = forward_backward_pipelining_without_interleaving_with_vlm_independent_parallelism
                elif args.zero_bubble_pipeline:
                    forward_backward_func = forward_backward_zero_bubble_pipelining_without_interleaving
                elif args.enable_forward_and_backward:
                    forward_backward_func = fab_forward_backward_pipelining_without_interleaving
                else:
                    forward_backward_func = forward_backward_pipelining_without_interleaving

            # train_batch in deepspeed/runtime/pipe/engine.py
            loss, losses_reduced = model[0].train_batch(data_iter=data_iterator, forward_backward_func=forward_backward_func,
                forward_step_func=forward_step_func, timers=timers, forward_only=False, teacher_model=teacher_model, extra_models=extra_models,
                encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func, lr_kwargs={'last_batch_iteration': args.iteration+1})
            vlm_optimizer_step()

            if args.activation_offloading:
                if args.virtual_pipeline_model_parallel_size is not None:
                    for vpp_rank in range(args.virtual_pipeline_model_parallel_size):
                        model[0].module[vpp_rank].offloading_ctx.reset()
                else:
                    model[0].module.offloading_ctx.reset()

            if args.profile_pipeline_parallel:
                timers = get_timers()
                pp_fwd_time = timers('forward-compute').elapsed(reset=False) * 1000
                pp_bwd_time = timers('backward-compute').elapsed(reset=False) * 1000
                pp_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

                def fp(elems, p):
                    return [round(e, p) for e in elems]

                ops = [torch.distributed.ReduceOp.MAX, torch.distributed.ReduceOp.MIN]
                for op in ops:
                    pp_info = torch.tensor([pp_fwd_time, pp_bwd_time, pp_mem], dtype=torch.float32, device='cuda')
                    torch.distributed.all_reduce(pp_info, op=op, group=mpu.get_tensor_model_parallel_group())
                    torch.distributed.all_reduce(pp_info, op=op, group=mpu.get_data_parallel_group())

                    pp_rank = mpu.get_pipeline_model_parallel_rank()
                    pp_size = mpu.get_pipeline_model_parallel_world_size()
                    all_pp_info = pp_info.new_empty((pp_size,) + pp_info.shape)
                    torch.distributed.all_gather_into_tensor(all_pp_info, pp_info, group=mpu.get_pipeline_model_parallel_group())

                    print_rank_0('pp fwd time: {}'.format(fp(all_pp_info[:, 0].tolist(), 2)))
                    print_rank_0('pp bwd time: {}'.format(fp(all_pp_info[:, 1].tolist(), 2)))
                    print_rank_0('pp mem: {}'.format(fp(all_pp_info[:, 2].tolist(), 2)))

        else:
            assert isinstance(model[0], deepspeed.PipelineEngine)
            loss, losses_reduced = model[0].train_batch(data_iter=data_iterator, lr_kwargs={'last_batch_iteration': args.iteration+1})
        grad_norm = model[0].get_global_grad_norm()
        if args.aux_moe:
            # Average loss across microbatches.
            # Then reduce moe loss across the pipeline group
            output_losses_reduced = {}
            if len(losses_reduced) == 0:
                assert args.mmp_encoder_parallel and mpu.is_mmp_encoder_stage(), "only mmp encoder stage has no losses_reduced"
                return losses_reduced, skipped_iter, grad_norm, num_zeros_in_grad
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                output_losses_reduced[key] = sum(losses_reduced_for_key) / get_num_microbatches()
            moe_losses_key = list(losses_reduced[0].keys())
            if 'lm loss' in moe_losses_key:
                moe_losses_key.remove('lm loss')
            if not args.use_mixed_training:
                for key in moe_losses_key:
                    if key in ['moe loss', 'z loss', 'exp capacity rate']:
                        torch.distributed.all_reduce(output_losses_reduced[key], group=mpu.get_pipeline_model_parallel_group())
                if args.log_all_moe_loss_to_tensorboard:
                    output_losses_reduced = get_all_moe_loss(moe_losses_key, output_losses_reduced)
            output_losses_reduced['lm loss'] = loss['lm loss']
            return output_losses_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        else:
            return loss, skipped_iter, grad_norm, num_zeros_in_grad


    # Set grad to zero.
    if not args.deepspeed:
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_ddp:
            for partition in model:
                partition.zero_grad_buffer()
        else:
            optimizer.zero_grad()

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            if args.mmp_encoder_parallel:
                return forward_backward_pipelining_with_interleaving_with_mmp_encoder_parallel
            elif args.enable_forward_and_backward:
                forward_backward_func = fab_forward_backward_pipelining_with_interleaving
            else:
                forward_backward_func = forward_backward_pipelining_with_interleaving
                if args.vlm_independent_parallelism:
                    forward_backward_func = forward_backward_pipelining_with_interleaving_with_vlm_independent_parallelism
                assert get_num_microbatches() % args.pipeline_model_parallel_size == 0, \
                    'number of microbatches is not divisible by pipeline-parallel ' \
                    'size when using interleaved schedule'
        else:
            if args.mmp_encoder_parallel:
                forward_backward_func = forward_backward_pipelining_without_interleaving_with_mmp_encoder_parallel
            elif args.vlm_independent_parallelism:
                forward_backward_func = forward_backward_pipelining_without_interleaving_with_vlm_independent_parallelism
            elif args.zero_bubble_parallel:
                forward_backward_func = forward_backward_zero_bubble_pipelining_without_interleaving
            elif args.enable_forward_and_backward:
                forward_backward_func = fab_forward_backward_pipelining_without_interleaving
            else:
                forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        if args.mmp_encoder_parallel:
            forward_backward_func = forward_backward_no_pipelining_with_mmp_encoder_parallel
        elif args.vlm_independent_parallelism:
            forward_backward_func = forward_backward_no_pipelining_with_vlm_independent_parallelism
        else:
            forward_backward_func = forward_backward_no_pipelining

    losses_reduced = forward_backward_func(
        forward_step_func, data_iterator, model,
        optimizer, timers, forward_only=False, teacher_model=teacher_model,
        encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)

    if args.activation_offloading:
        if args.virtual_pipeline_model_parallel_size is not None:
            for vpp_rank in range(args.virtual_pipeline_model_parallel_size):
                model[0].module[vpp_rank].offloading_ctx.reset()
        else:
            model[0].module.offloading_ctx.reset()

    # All-reduce if needed.
    if not args.deepspeed and args.DDP_impl == 'local':
        timers('backward-params-all-reduce', log_level=2).start()
        for model_module in model:
            model_module.allreduce_gradients()
        timers('backward-params-all-reduce').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers('backward-embedding-all-reduce', log_level=2).start()
    if not args.deepspeed:
        if mpu.is_rank_in_embedding_group(ignore_virtual=True) and \
                mpu.get_pipeline_model_parallel_world_size() > 1:
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                unwrapped_model = model[0]
            elif mpu.is_pipeline_last_stage(ignore_virtual=True):
                unwrapped_model = model[-1]
            else:
                unwrapped_model = model[0]
            unwrapped_model = unwrap_model(
                unwrapped_model, (torchDDP, LocalDDP, Float16Module))

            if unwrapped_model.share_word_embeddings:
                word_embeddings_weight = unwrapped_model.word_embeddings_weight()
                if args.DDP_impl == 'local':
                    grad = word_embeddings_weight.main_grad
                else:
                    grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
        if mpu.is_rank_in_position_embedding_group() and \
            mpu.get_pipeline_model_parallel_world_size() > 1 and \
            args.pipeline_model_parallel_split_rank is not None:
            unwrapped_model = model[0]
            unwrapped_model = unwrap_model(unwrapped_model, (torchDDP, LocalDDP, Float16Module))
            grad = unwrapped_model.language_model.embedding.position_embeddings.weight.main_grad
            torch.distributed.all_reduce(grad, group=mpu.get_position_embedding_group())
    timers('backward-embedding-all-reduce').stop()

    # Update parameters.
    timers('optimizer', log_level=2).start()
    if args.deepspeed:
        increment = get_num_microbatches() * \
                    get_current_micro_batch_size() * \
                    args.data_parallel_size
        model[0].step(lr_kwargs={'last_batch_iteration': args.iteration+1})
        update_successful = model[0].was_step_applied()
    else:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()
    vit_update_successful = vlm_optimizer_step()

    # Update learning rate.
    if args.deepspeed:
        skipped_iter = 0
        grad_norm = model[0].get_global_grad_norm()
        num_zeros_in_grad = None

        loss_reduced = {}
        if len(losses_reduced) == 0:
            assert args.mmp_encoder_parallel and mpu.is_mmp_encoder_stage(), "only mmp encoder stage has no losses_reduced"
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        if max(args.num_experts) > 1 and args.log_all_moe_loss_to_tensorboard:
            moe_losses_key = list(losses_reduced[0].keys())
            moe_losses_key.remove('lm loss')
            loss_reduced = get_all_moe_loss(moe_losses_key, loss_reduced)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    else:
        if update_successful:
            increment = get_num_microbatches() * \
                        get_current_micro_batch_size() * \
                        args.data_parallel_size
            lr_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        if args.moe:
            # Average loss across microbatches.
            # Then reduce moe loss across the pipeline group.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / get_num_microbatches()
            moe_losses_key = list(losses_reduced[0].keys())
            moe_losses_key.remove('lm loss')
            for key in moe_losses_key:
                torch.distributed.all_reduce(loss_reduced[key], group=mpu.get_pipeline_model_parallel_group())
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 model=None, optimizer=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if key == 'lm loss' or 'mos loss':
            loss_dict[key] = loss_dict[key].mean()
            # TODO(yuanbopeng) need confirm
            #loss_dict[key] = average_losses_across_data_parallel_group([loss_dict[key]])[0]
            if key == 'lm loss' and args.context_parallel_size > 1:
                # recover lm loss
                loss_dict[key] /= args.context_parallel_size
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key].detach()
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('p2p')
    add_to_logging('moe')
    add_to_logging('forward-compute')
    add_to_logging('chunk0_forward')
    add_to_logging('chunk1_forward')
    add_to_logging('backward-compute')
    add_to_logging('chunk0_backward')
    add_to_logging('chunk1_backward')
    add_to_logging('forward-backward-send-forward-backward-recv')
    add_to_logging('forward-backward-send-forward-backward-recv-v')
    add_to_logging('forward-recv')
    add_to_logging('forward-send')
    add_to_logging('backward-recv')
    add_to_logging('backward-send')
    add_to_logging('backward-send')
    add_to_logging('forward-send-forward-recv')
    add_to_logging('forward-send-backward-recv')
    add_to_logging('backward-send-forward-recv')
    add_to_logging('backward-send-backward-recv')
    add_to_logging('backward-params-all-reduce')
    add_to_logging('backward-embedding-all-reduce')
    add_to_logging('optimizer-copy-to-main-grad')
    add_to_logging('optimizer-unscale-and-check-inf')
    add_to_logging('optimizer-clip-main-grad')
    add_to_logging('optimizer-copy-main-to-model-params')
    add_to_logging('optimizer')
    add_to_logging('batch-generator')
    add_to_logging('batch-generator-0')
    add_to_logging('batch-generator-1')
    add_to_logging('dataloader')
    add_to_logging('data-process')
    add_to_logging('vit-forward')
    add_to_logging('vit-backward')
    add_to_logging('vit-perceive')
    add_to_logging('vit-optimizer')
    add_to_logging('audio-encoder-forward')
    add_to_logging('fix_emb.fwd')
    add_to_logging('var_emb.fwd')
    add_to_logging('mixed_mlp_moe')
    add_to_logging('falltoall')
    add_to_logging('salltoall')

    # Calculate batch size.
    batch_size = get_current_micro_batch_size() * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0) and \
       is_last_rank():
        timers('writing-tensorboard', log_level=2).start()
        writer.add_scalar('steps-vs-samples/y=steps,x=samples', iteration, args.consumed_train_samples)
        writer.add_scalar('steps-vs-samples/y=samples,x=steps', args.consumed_train_samples, iteration)
        writer.add_scalar('steps-vs-tokens/y=steps,x=tokens', iteration, args.consumed_train_tokens)
        writer.add_scalar('steps-vs-tokens/y=tokens,x=steps', args.consumed_train_tokens, iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate/learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate/learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            writer.add_scalar('learning-rate/learning-rate vs tokens', learning_rate,
                              args.consumed_train_tokens)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size/batch-size', batch_size, iteration)
            writer.add_scalar('batch-size/batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            if not key.startswith("layer"):
                writer.add_scalar(f"lm-loss-training/{key}", loss_dict[key], iteration)
                writer.add_scalar(f"lm-loss-training/{key}" + ' vs samples', loss_dict[key],
                                args.consumed_train_samples)
                writer.add_scalar(f"lm-loss-training/{key}" + ' vs tokens', loss_dict[key],
                                args.consumed_train_tokens)
        if args.log_all_moe_loss_to_tensorboard:
            for key in loss_dict:
                if key.startswith("layer"):
                    writer.add_scalar(f"moe-loss-training/{key}", loss_dict[key], iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale/loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale/loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            writer.add_scalar('loss-scale/loss-scale vs tokens', loss_scale,
                              args.consumed_train_tokens)
        if grad_norm is not None:
            writer.add_scalar('grad-norm/grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm/grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            writer.add_scalar('grad-norm/grad-norm vs tokens', grad_norm,
                              args.consumed_train_tokens)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros/num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros/num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            writer.add_scalar('num-zeros/num-zeros vs tokens', num_zeros_in_grad,
                              args.consumed_train_tokens)
        if params_norm is not None:
            writer.add_scalar('params-norm/params-norm', params_norm, iteration)
            writer.add_scalar('params-norm/params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            writer.add_scalar('params-norm/params-norm vs tokens', params_norm,
                              args.consumed_train_tokens)
        if args.curriculum_learning:
            writer.add_scalar('curriculum_seqlen', args.curriculum_seqlen,
                              iteration)
        if args.log_timers_to_tensorboard:
            timers.write(timers_to_log, writer, iteration,
                         normalizer=total_iterations)
        timers('writing-tensorboard').stop()

    if iteration % args.tensorboard_log_interval == 0:
        timers('writing-tensorboard', log_level=2).start()
        # This logging write various optimizer states to tensorboard. This
        # feature may consume extra GPU memory thus is set at false by default.
        if args.log_optimizer_states_to_tensorboard and optimizer is not None:
            opt_stats = [0.0] * 8
            opt_stats_2 = [0.0] * 4
            for _, group in enumerate(optimizer.param_groups):
                for _, param in enumerate(group['params']):
                    opt_stats[0] += (torch.norm(optimizer.state[param]['exp_avg_sq']).item())**2
                    opt_stats[1] += (torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt()).item())**2
                    opt_stats[2] += (torch.norm(optimizer.state[param]['exp_avg']).item())**2
                    opt_stats[3] += (torch.norm(param).item())**2
                    opt_stats[4] += torch.norm(optimizer.state[param]['exp_avg_sq'],p=1).item()
                    opt_stats[5] += torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt(),p=1).item()
                    opt_stats[6] += torch.norm(optimizer.state[param]['exp_avg'],p=1).item()
                    opt_stats[7] += torch.norm(param,p=1).item()
                    opt_stats_2[0] = max(opt_stats_2[0], abs(optimizer.state[param]['exp_avg_sq'].max().item()), abs(optimizer.state[param]['exp_avg_sq'].min().item()))
                    opt_stats_2[1] = max(opt_stats_2[1], optimizer.state[param]['exp_avg_sq'].sqrt().abs_().max().item())
                    opt_stats_2[2] = max(opt_stats_2[2], abs(optimizer.state[param]['exp_avg'].max().item()), abs(optimizer.state[param]['exp_avg'].min().item()))
                    opt_stats_2[3] = max(opt_stats_2[3], abs(param.max().item()), abs(param.min().item()))
            # print('step {} rank {} before sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if args.zero_stage > 0:
                # ZeRO partiions optimizer states
                opt_stats = torch.cuda.FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_data_parallel_group())
                opt_stats_2 = torch.cuda.FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_data_parallel_group())

            if args.tensor_model_parallel_size > 1:
                opt_stats = torch.cuda.FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_tensor_model_parallel_group())
                opt_stats_2 = torch.cuda.FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_tensor_model_parallel_group())

            if args.pipeline_model_parallel_size > 1:
                opt_stats = torch.cuda.FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_pipeline_model_parallel_group())
                opt_stats_2 = torch.cuda.FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_pipeline_model_parallel_group())

            # print('step {} rank {} after sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if writer and is_last_rank():
                writer.add_scalar('optimizer/variance_l2 vs tokens', opt_stats[0]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l2 vs tokens', opt_stats[1]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l2 vs tokens', opt_stats[2]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l2 vs tokens', opt_stats[3]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_l1 vs tokens', opt_stats[4], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l1 vs tokens', opt_stats[5], args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l1 vs tokens', opt_stats[6], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l1 vs tokens', opt_stats[7], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_abs_max vs tokens', opt_stats_2[0], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_abs_max vs tokens', opt_stats_2[1], args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_abs_max vs tokens', opt_stats_2[2], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_abs_max vs tokens', opt_stats_2[3], args.consumed_train_tokens)

                writer.add_scalar('optimizer/variance_l2', opt_stats[0]**0.5, iteration)
                writer.add_scalar('optimizer/variance_sqrt_l2', opt_stats[1]**0.5, iteration)
                writer.add_scalar('optimizer/momentum_l2', opt_stats[2]**0.5, iteration)
                writer.add_scalar('optimizer/weight_l2', opt_stats[3]**0.5, iteration)
                writer.add_scalar('optimizer/variance_l1', opt_stats[4], iteration)
                writer.add_scalar('optimizer/variance_sqrt_l1', opt_stats[5], iteration)
                writer.add_scalar('optimizer/momentum_l1', opt_stats[6], iteration)
                writer.add_scalar('optimizer/weight_l1', opt_stats[7], iteration)
                writer.add_scalar('optimizer/variance_abs_max', opt_stats_2[0], iteration)
                writer.add_scalar('optimizer/variance_sqrt_abs_max', opt_stats_2[1], iteration)
                writer.add_scalar('optimizer/momentum_abs_max', opt_stats_2[2], iteration)
                writer.add_scalar('optimizer/weight_abs_max', opt_stats_2[3], iteration)
        timers('writing-tensorboard').stop()
    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed()
        total_time = timers('total-time').elapsed(False)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        assert iteration > args.init_iteration, f"iteration {iteration} must large than init_iteration {args.init_iteration}"
        avg_elapsed_time_per_iteration = total_time / (iteration - args.init_iteration)
        if iteration - args.init_iteration > 2:
            args.avg_is_valid = True
        # only the last rank process has a non-None _GLOBAL_TENSORBOARD_WRITER
        if writer and is_last_rank():
            if args.log_timers_to_tensorboard:
                timers('writing-tensorboard', log_level=2).start()
                writer.add_scalar('iteration-time/iteration-time',
                                  elapsed_time_per_iteration, iteration)
                writer.add_scalar('iteration-time/iteration-time vs samples',
                                  elapsed_time_per_iteration, args.consumed_train_samples)
                writer.add_scalar('iteration-time/iteration-time vs tokens',
                                  elapsed_time_per_iteration, args.consumed_train_tokens)
                writer.add_scalar('iteration-time/avg-iteration-time',
                                  avg_elapsed_time_per_iteration, iteration)
                writer.add_scalar('iteration-time/avg-iteration-time vs samples',
                                  avg_elapsed_time_per_iteration, args.consumed_train_samples)
                writer.add_scalar('iteration-time/avg-iteration-time vs tokens',
                                  avg_elapsed_time_per_iteration, args.consumed_train_tokens)
                timers('writing-tensorboard').stop()
        if writer and is_first_rank():
            writer.add_scalar('vit/vit_forward', timers('vit-forward').elapsed(False) / total_iterations, iteration)
            writer.add_scalar('vit/var_emb.fwd', timers('var_emb.fwd').elapsed(False) / total_iterations, iteration)
            writer.add_scalar('vit/vit_perceive', timers('vit-perceive').elapsed(False) / total_iterations, iteration)
            writer.add_scalar('vit/vit_optimizer', timers('vit-optimizer').elapsed(False) / total_iterations, iteration)
            writer.add_scalar('vit/vit_backward', timers('vit-backward').elapsed(False) / total_iterations, iteration)


        add_to_logging('writing-tensorboard')
        log_string = 'rank {} | iteration {:8d}/{:8d} |'.format(torch.distributed.get_rank(),
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' consumed tokens: {:12d} |'.format(
            args.consumed_train_tokens)
        log_string += ' elapsed time this iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' avg elapsed time per iteration (ms): {:.1f} |'.format(
            avg_elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)

        # add lm/t2i/mm... consumed samples
        gemini_trainer = get_gemini_trainer()
        for key in gemini_trainer.ss.epoch_consumed_samples.keys():
            value = gemini_trainer.ss.epoch_consumed_samples[key]
            log_string += (key + ' consumed samples: {:12d} |'.format(value))

        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                # if avg > 0.0 or key in ["better", "worse", "gap", "ctr_loss", "sft_loss", "dpo_loss", "log_ratio"]:
                log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        if args.curriculum_learning:
            log_string += ' curriculum seqlen: {:5d} |'.format(args.curriculum_seqlen)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        if args.log_lr_of_all_param_groups:
            log_string += ' lr of param_groups:'
            for param_group in optimizer.param_groups:
                log_string += ' {}={:.3E}'.format(param_group['name'], param_group['lr'])
            log_string += " |"
        mfu = calc_mfu(batch_size, elapsed_time)
        log_string += ' mfu: {:.4f} |'.format(mfu)
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval, message=log_string)
        flops_calculator(model, args, elapsed_time)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler, ckpt_nums):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    torch.distributed.barrier()
    args = get_args()

    # save trainer_ptm_wrapper.ss TODO: implement the serialization function of ss, instead of calling to_dict before save_checkpoint. by youngfyang
    gemini_trainer = get_gemini_trainer()
    args.scalar_states = gemini_trainer.ss.to_dict()

    save_tensors_mode=args.save_tensors_mode if args.save_ckpt_with_tensors else None
    timers('save-checkpoint', log_level=0).start()
    save_checkpoint(iteration, model, optimizer, lr_scheduler, ckpt_nums, save_tensors_mode=save_tensors_mode)
    save_vit_checkpoint(iteration, save_tensors_mode)
    torch.distributed.barrier()
    timers('save-checkpoint').stop()
    timers.log(['save-checkpoint'])

def get_profile_args():
    args = get_args()
    profile_enable = args.profile_enable
    schedule_wait = args.profile_schedule_wait
    schedule_warmup = args.profile_schedule_warmup
    schedule_active = args.profile_schedule_active
    schedule_repeat = args.profile_schedule_repeat
    tensorboard_trace_handler_dir = os.path.join(args.profile_tensorboard_trace_handler_dir, str(torch.distributed.get_rank()))
    profile_memory = args.profile_memory
    with_stack = args.profile_with_stack
    return profile_enable, schedule_wait, schedule_warmup, schedule_active, schedule_repeat, tensorboard_trace_handler_dir, profile_memory, with_stack

def torch_version_correct(profile_torch_version):
    """
    profile_torch_version: 1.10.0
    """
    cur_v = torch.__version__
    cur_v_list = cur_v.split('.')
    assert len(cur_v_list) >= 2
    cur_v_major = int(cur_v_list[0])
    cur_v_minor = int(cur_v_list[1])

    given_v_list = profile_torch_version.split('.')
    assert len(given_v_list) >= 2
    given_v_major = int(given_v_list[0])
    given_v_minor = int(given_v_list[1])

    return cur_v_major > given_v_major or \
        (cur_v_major == given_v_major and cur_v_minor >= given_v_minor)


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator, teacher_model=None, extra_models=None, eval_forward_step_func=None,
          for_stream_ds=None, custom_inner_while_train_func=None, encoder_llm_comm_tensor_shape_func=None):
    """Train the model function."""
    args = get_args()
    timers = get_timers()
    gemini_trainer = get_gemini_trainer() # for update gemini_trainer.ss.epoch_consumed_samples
    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    set_vit_model_state(True)
    for model_module in model:
        model_module.train()

    if args.enable_model_precision_tracker:
        from ptm.utils import initailize_model_precision_trakcer

        initailize_model_precision_trakcer()
    # Tracking loss.
    total_loss_dict = {}

    timers('interval-time', log_level=0).start()
    timers('total-time', log_level=0).start()
    print_datetime('before the start of training step')

    report_memory_flag = True
    profile_enable, schedule_wait, schedule_warmup, schedule_active, schedule_repeat, \
            tensorboard_trace_handler_dir, profile_memory, with_stack = get_profile_args()
    profile_torch_version = '1.10.0'
    correct = torch_version_correct(profile_torch_version)
    if profile_enable:
        assert correct , "torch version(%s) must greater equal %s" % (torch.__version__, profile_torch_version)
        profile_enable = profile_enable and correct

    def profiler_decorator(train_func):
        def wrapper(profiler_ctx):
            if not profile_enable:
                train_func(None)
            else:
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(
                            wait = schedule_wait,
                            warmup = schedule_warmup,
                            active = schedule_active,
                            repeat = schedule_repeat
                        ),
                        profile_memory=profile_memory,
                        with_stack=with_stack,
                        on_trace_ready = torch.profiler.tensorboard_trace_handler(tensorboard_trace_handler_dir),
                    ) as profiler:
                    train_func(profiler)

        return wrapper

    # Iterations.
    iteration = args.iteration
    if args.eval_save_model_key == None:
        best_score = None
    else:
        if args.eval_save_model_func == "greater":
            best_score = 0.0
        else:
            best_score = 10000.0

    if iteration < args.train_iters and (args.train_tokens is None or \
        args.consumed_train_tokens < args.train_tokens):
        import os
        if os.getenv('PROFILE'):
            if iteration == 21 and (os.getenv('PROFILE')=='all' or torch.distributed.get_rank() == int(os.getenv('PROFILE'))):
                print("Rank %d nsys profiling - start." % torch.distributed.get_rank())
                torch.cuda.cudart().cudaProfilerStart()
            if iteration == 26:
                if (os.getenv('PROFILE')=='all' or torch.distributed.get_rank() == int(os.getenv('PROFILE'))):
                    print("Rank %d nsys profiling - complete." % torch.distributed.get_rank())
                    torch.cuda.cudart().cudaProfilerStop()
                torch.distributed.barrier()
                exit()
        update_num_microbatches(args.consumed_train_samples)
        args.num_micro_batches = get_num_microbatches()
        args.current_global_batch_size = get_current_global_batch_size()
        if args.deepspeed:
            # inform deepspeed of any batch size changes
            global_batch_size = mpu.get_data_parallel_world_size() * \
                                get_current_micro_batch_size() * \
                                get_num_microbatches()
            model[0].set_train_batch_size(global_batch_size)

    if args.use_stream_dataset and for_stream_ds[0]:
        train_data_iterator = iter(build_pretraining_data_loader(
            for_stream_ds[0], args.consumed_train_samples - args.consumed_train_dataset_len))
    if args.use_stream_dataset and for_stream_ds[1]:
        valid_data_iterator = iter(build_pretraining_data_loader(
            for_stream_ds[1], max(args.consumed_valid_samples - args.consumed_train_dataset_len, 0)))

    if args.use_iteration_dataset and for_stream_ds[0] is not None:
        train_data_iterator = iter(for_stream_ds[0])
    if args.use_iteration_dataset and for_stream_ds[1] is not None:
        valid_data_iterator = iter(for_stream_ds[1])

    @profiler_decorator
    def inner_while_train(profiler_ctx):
        nonlocal iteration, report_memory_flag, best_score, train_data_iterator, valid_data_iterator
        import gc
        gc.set_threshold(7000, 1000, 1000)
        while iteration < args.train_iters and (args.train_tokens is None or \
            args.consumed_train_tokens < args.train_tokens):
            _call_callback_hooks(model[0].module, "on_train_batch_start", iteration+1)
            from megatron.pipeline_recorder import PipelineRecorderSwitch
            if iteration < 3: # ignore former steps
                PipelineRecorderSwitch.turn_off()
            else:
                PipelineRecorderSwitch.turn_on()
            update_num_microbatches(args.consumed_train_samples)
            args.num_micro_batches = get_num_microbatches()
            args.current_global_batch_size = get_current_global_batch_size()
            if args.deepspeed:
                # inform deepspeed of any batch size changes
                global_batch_size = mpu.get_data_parallel_world_size() * \
                                    get_current_micro_batch_size() * \
                                    get_num_microbatches()
                model[0].set_train_batch_size(global_batch_size)

            if args.curriculum_learning and not args.no_pipeline_parallel:
                args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                        args.iteration + 1)

            # broadcast a zero size tensor to indicate starting of step
            start_flag_tensor = torch.cuda.FloatTensor([])
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(start_flag_tensor, 0, async_op=True)

            # record the current step's lr for log
            training_log_lr = optimizer.param_groups[0]['lr']

            torch.compiler.cudagraph_mark_step_begin()
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(forward_step_func,
                           train_data_iterator,
                           model,
                           optimizer,
                           lr_scheduler,
                           teacher_model=teacher_model,
                           extra_models=extra_models,
                           encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)

            if loss_dict == None:
                # encoder_stage does not maintain loss
                assert args.mmp_encoder_parallel and mpu.is_mmp_encoder_stage()
                loss_dict = {}
            args.is_first_iter = False

            iteration += 1
            args.iteration = iteration
            gemini_trainer.ss.current_run_update_steps += 1
            new_samples = mpu.get_data_parallel_world_size() * \
                                           get_current_micro_batch_size() * \
                                           get_num_microbatches()
            args.consumed_train_samples += new_samples
            if args.curriculum_learning:
                args.consumed_train_tokens += new_samples * args.curriculum_seqlen
            else:
                args.consumed_train_tokens += new_samples * args.seq_length

            _call_callback_hooks(model[0].module, "on_train_batch_end", iteration, loss_dict)

            # update ss.epoch_consumed_samples for continue train
            all_gather_consumed_samples = [None for _ in range(mpu.get_data_parallel_world_size())]  
            torch.distributed.all_gather_object(all_gather_consumed_samples, args.consumed_samples_per_step, group=mpu.get_data_parallel_group())
            for item in all_gather_consumed_samples:
                for key,value in item.items():
                    gemini_trainer.ss.epoch_consumed_samples.setdefault(key, 0)
                    gemini_trainer.ss.epoch_consumed_samples[key] += value
            args.consumed_samples_per_step = {} # clear for every step

            # For fp8 and bf16 mixed training phase
            if get_train_precision() == 'fp8' and iteration >= args.fp8_ratio * args.train_iters:
                print_rank_0(f'[FP8] current iteration {iteration}, change fp8 training to bf16 traing ...')
                set_train_precision('bf16')

            clear_te_cache()

            # Logging.
            # Is loss_scale valid? need to discuss.
            if args.fp16:
                if args.deepspeed:
                    loss_scale = model[0].optimizer.cur_scale
                else:
                    loss_scale = optimizer.get_loss_scale().item()
            else:
                loss_scale = 1.0
            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(model)
            report_memory_flag = training_log(loss_dict, total_loss_dict,
                                              training_log_lr,
                                              iteration, loss_scale,
                                              report_memory_flag, skipped_iter,
                                              grad_norm, params_norm, num_zeros_in_grad,
                                              model, optimizer)     # where megatron elapsed_time calculated

            # For light timing log
            if not args.disable_light_timing and iteration >= args.timing_schedule_wait and args.timing_schedule_repeat >= 0:
                if args.timing_schedule_repeat == 0:
                    # reset to origin at the end
                    timers.reset_log_level(args.timing_log_level)
                    args.timing_schedule_repeat -= 1
                elif ((iteration - args.timing_schedule_wait) % args.timing_schedule_interval) < args.timing_schedule_iters:
                    timers.reset_log_level(2)
                    args.timing_schedule_repeat -= 1
                else:
                    timers.reset_log_level(args.timing_log_level)

            # Autoresume
            if args.adlr_autoresume and \
               (iteration % args.adlr_autoresume_interval == 0):
                check_adlr_autoresume_termination(iteration, model, optimizer,
                                                  lr_scheduler)

            #stream dataset check
            if args.use_stream_dataset and iteration >= args.train_iters:
               update_dataset = check_stream_data_prefix(args.data_path[0], for_stream_ds, check_end=True)
               while update_dataset is False:
                   print_rank_0('> data use out checking new stream data ...')
                   time.sleep(30)
                   update_dataset = check_stream_data_prefix(args.data_path[0], for_stream_ds, check_end=True)
               if update_dataset is True:
                   if for_stream_ds[0]:
                       train_data_iterator = iter(build_pretraining_data_loader(
                           for_stream_ds[0], args.consumed_train_samples - args.consumed_train_dataset_len))
                       lazy_gc(iteration=iteration)
                   if for_stream_ds[1]:
                       valid_data_iterator = iter(build_pretraining_data_loader(
                           for_stream_ds[1], max(args.consumed_valid_samples - args.consumed_train_dataset_len, 0)))
                       lazy_gc(iteration=iteration)
                   if for_stream_ds[0]:
                       if args.consumed_train_dataset >0:
                           remove_consume_stream_data(args.data_path[0])
                           args.consumed_train_dataset = 0
               else:
                   print_rank_0('>[use_stream_dataset] data consumed out ...')
                   break

            elif args.use_stream_dataset and iteration % args.stream_check_interval == 0:
               update_dataset = check_stream_data_prefix(args.data_path[0], for_stream_ds)
               if update_dataset and for_stream_ds[0]:
                   train_data_iterator = iter(build_pretraining_data_loader(
                       for_stream_ds[0], args.consumed_train_samples - args.consumed_train_dataset_len))
                   lazy_gc(iteration=iteration)
               if update_dataset and for_stream_ds[1]:
                   valid_data_iterator = iter(build_pretraining_data_loader(
                       for_stream_ds[1], max(args.consumed_valid_samples - args.consumed_train_dataset_len, 0)))
                   lazy_gc(iteration=iteration)
               if update_dataset and for_stream_ds[0]:
                   if args.consumed_train_dataset >0:
                       remove_consume_stream_data(args.data_path[0])
                       args.consumed_train_dataset = 0

            if valid_data_iterator is not None and args.use_multi_validation_dataset and \
                args.virtual_pipeline_model_parallel_size:
                valid_data_iterator = [list(t) for t in zip(*valid_data_iterator)]

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0 and \
               args.do_valid:
                _call_callback_hooks(model[0].module, "on_validation_batch_start", iteration, dataloader_idx=0)
                prefix = 'iteration {}'.format(iteration)
                if args.eval_save_model_key != None:
                    best_score = evaluate_and_print_results(prefix, eval_forward_step_func,
                                                            valid_data_iterator, model,
                                                            iteration, False, best_score=best_score, teacher_model=teacher_model,
                                                            encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)
                else:
                    evaluate_and_print_results(prefix, eval_forward_step_func,
                                            valid_data_iterator, model,
                                            iteration, False, teacher_model=teacher_model,
                                            encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)
                _call_callback_hooks(model[0].module, "on_validation_batch_end", iteration, {}, dataloader_idx=0)

            # Checkpointing
            saved_checkpoint = False
            if args.save and args.save_interval and \
               iteration % args.save_interval == 0 and args.eval_save_model_key == None:
                if args.use_lora:
                    convert_lora_to_linear_layer(model[0].module)
                save_checkpoint_and_time(iteration, model, optimizer,
                                         lr_scheduler, args.ckpt_nums)
                if args.use_lora:
                    unfuse_lora_from_linear_layer(model[0].module)
                saved_checkpoint = True

            # Exiting based on duration
            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.cuda.IntTensor(
                    [train_time > args.exit_duration_in_mins])
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    if (not saved_checkpoint or args.use_lora) and args.eval_save_model_key == None:
                        if args.use_lora:
                            convert_lora_to_linear_layer(model[0].module)
                        save_checkpoint_and_time(iteration, model, optimizer,
                                                 lr_scheduler, args.ckpt_nums)
                    print_datetime('exiting program after {} minutes'.format(train_time))
                    sys.exit()

            # Exiting based on iterations
            if args.exit_interval and iteration % args.exit_interval == 0:
                if (not saved_checkpoint or args.use_lora) and args.eval_save_model_key == None:
                    if args.use_lora:
                        convert_lora_to_linear_layer(model[0].module)
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             lr_scheduler, args.ckpt_nums)
                torch.distributed.barrier()
                print_datetime('exiting program at iteration {}'.format(iteration))
                sys.exit()
            # args.profile_enable AND torch version >= '1.10.0'
            if profile_enable:
                profiler_ctx.step()

    if custom_inner_while_train_func:
        inner_while_train = custom_inner_while_train_func
    inner_while_train(None)

    return iteration


def evaluate(forward_step_func, data_iterator, model, verbose=False, teacher_model=None, encoder_llm_comm_tensor_shape_func=None):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()
    set_vit_model_state(False)

    if args.curriculum_learning and not args.no_pipeline_parallel:
        # When curriculum learning is used with pipeline parallelism, we need
        # this logic to ensure that the eval data is not truncated. If there
        # is a seqlen change due to that, we need to call
        # reset_activation_shape() to reset some buffers in deepspeed pipeline
        # engine.
        if args.curriculum_seqlen < args.seq_length:
            args.curriculum_seqlen = args.seq_length
            model[0].reset_activation_shape()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            if mpu.get_pipeline_model_parallel_world_size() > 1:
                if args.virtual_pipeline_model_parallel_size is not None:
                    if args.mmp_encoder_parallel:
                        forward_backward_func = forward_backward_pipelining_with_interleaving_with_mmp_encoder_parallel
                    elif args.vlm_independent_parallelism:
                        forward_backward_func = forward_backward_pipelining_with_interleaving_with_vlm_independent_parallelism
                    elif args.enable_forward_and_backward:
                        forward_backward_func = fab_forward_backward_pipelining_with_interleaving
                    else:
                        forward_backward_func = forward_backward_pipelining_with_interleaving
                else:
                    if args.mmp_encoder_parallel:
                        forward_backward_func = forward_backward_pipelining_without_interleaving_with_mmp_encoder_parallel
                    elif args.vlm_independent_parallelism:
                        forward_backward_func = forward_backward_pipelining_without_interleaving_with_vlm_independent_parallelism
                    elif args.zero_bubble_pipeline:
                        forward_backward_func = forward_backward_zero_bubble_pipelining_without_interleaving
                    elif args.enable_forward_and_backward:
                        forward_backward_func = fab_forward_backward_pipelining_without_interleaving
                    else:
                        forward_backward_func = forward_backward_pipelining_without_interleaving
            else:
                if args.mmp_encoder_parallel:
                    forward_backward_func = forward_backward_no_pipelining_with_mmp_encoder_parallel
                elif args.vlm_independent_parallelism:
                    forward_backward_func = forward_backward_no_pipelining_with_vlm_independent_parallelism
                elif args.enable_forward_and_backward:
                    forward_backward_func = fab_forward_backward_no_pipelining
                else:
                    forward_backward_func = forward_backward_no_pipelining

            if args.deepspeed and args.ds_pipeline_enabled:
                if not args.mt_pipeline_parallel:
                    # DeepSpeed uses eval_batch() and already aggregates losses.
                    assert isinstance(model, list) and len(model) == 1
                    loss = model[0].eval_batch(data_iterator, timers=timers)
                    loss_dicts = [{'lm loss' : loss}] * get_num_microbatches()
                else:
                    loss_dicts = model[0].eval_batch(data_iter=data_iterator, forward_backward_func=forward_backward_func, \
                                  forward_step_func=forward_step_func, timers=timers, forward_only=True, teacher_model=teacher_model, encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)
            else:
                loss_dicts = forward_backward_func(
                    forward_step_func, data_iterator, model, optimizer=None,
                    timers=None, forward_only=True, teacher_model=teacher_model, encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)

            # Reduce across processes.
            for loss_dict in loss_dicts:
                for key in loss_dict:
                    if 'moe' not in key:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * get_current_micro_batch_size() \
                                           * get_num_microbatches()
    # Move model back to the train mode.
    for model_module in model:
        model_module.train()
    set_vit_model_state(True)

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()
        if args.aux_moe:
            if key in ['moe loss', 'z loss', 'exp capacity rate']:
                torch.distributed.all_reduce(total_loss_dict[key], group=mpu.get_pipeline_model_parallel_group())

    if args.curriculum_learning and not args.no_pipeline_parallel:
        # roll back to actual curriculum seqlen at the end of eval.
        args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
            args.iteration + 1)
        if args.curriculum_seqlen < args.seq_length:
            model[0].reset_activation_shape()

    return total_loss_dict


def evaluate_and_print_results(prefix, eval_forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False, teacher_model=None, **kwargs):
    """Helper function to evaluate and dump results on screen."""
    # reduce(MIN op) a zero size tensor to indicate starting of evaluate for profiling tool
    eval_flag_tensor = torch.cuda.FloatTensor([])
    if torch.distributed.is_initialized():
        torch.distributed.reduce(eval_flag_tensor, 0, op=torch.distributed.ReduceOp.MIN, async_op=True)

    args = get_args()
    writer = get_tensorboard_writer()
    encoder_llm_comm_tensor_shape_func = kwargs.get("encoder_llm_comm_tensor_shape_func", None)

    set_validate_batches_states(True)
    if isinstance(eval_forward_step_func, dict):
        eval_forward_step_func_list = []
        for key in eval_forward_step_func:
            eval_forward_step_func_list += [eval_forward_step_func[key]] * key
        eval_forward_step_func = eval_forward_step_func_list

    if args.use_multi_validation_dataset is None:
        assert not isinstance(eval_forward_step_func, list)
        total_loss_dict = evaluate(eval_forward_step_func, data_iterator, model, verbose, teacher_model=teacher_model, encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)
    else:
        if not isinstance(eval_forward_step_func, list):
            eval_forward_step_func_list = [eval_forward_step_func] * len(data_iterator)
            eval_forward_step_func = eval_forward_step_func_list
        assert len(data_iterator) == len(eval_forward_step_func)
        total_loss_dict = {}
        for i in range(len(data_iterator)):
            this_dataset_loss_dict = evaluate(eval_forward_step_func[i], data_iterator[i], model, verbose, teacher_model=teacher_model, encoder_llm_comm_tensor_shape_func=encoder_llm_comm_tensor_shape_func)
            this_dataset_name = args.use_multi_validation_dataset[i].split(":")[0]
            for key in this_dataset_loss_dict:
                full_key_name = f"{this_dataset_name}_{key}"
                total_loss_dict[full_key_name] = this_dataset_loss_dict[key]

    set_validate_batches_states(False)
    string = ' validation result at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        if "lm" in key:
            ppl = math.exp(min(20, total_loss_dict[key].item()))
            string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and is_last_rank():
            writer.add_scalar(f'lm-loss-validation/{key} validation',
                            total_loss_dict[key].item(),
                            iteration)
            writer.add_scalar(f'lm-loss-validation/{key} validation vs samples',
                            total_loss_dict[key].item(),
                            args.consumed_train_samples)
            writer.add_scalar(f'lm-loss-validation/{key} validation vs tokens',
                            total_loss_dict[key].item(),
                            args.consumed_train_tokens)
            if "lm" in key and args.log_validation_ppl_to_tensorboard:
                writer.add_scalar(f'lm-loss-validation/{key} validation ppl', ppl,
                                iteration)
                writer.add_scalar(f'lm-loss-validation/{key} validation ppl vs samples',
                                ppl, args.consumed_train_samples)
                writer.add_scalar(f'lm-loss-validation/{key} validation ppl vs tokens',
                                ppl, args.consumed_train_tokens)


    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)

    best_score = kwargs.get("best_score", None)
    best_score_save_flag = False
    new_iter_best_score = 0.0
    if args.eval_save_model_key != None and best_score != None:
        if args.eval_save_model_key in total_loss_dict:
            new_iter_best_score = total_loss_dict[args.eval_save_model_key].item()
            best_score_save_flag = (args.eval_save_model_func == 'greater' and new_iter_best_score > best_score) or \
                (args.eval_save_model_func == 'less' and new_iter_best_score < best_score)
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                flag_tensor = torch.cuda.FloatTensor([best_score_save_flag])
                torch.distributed.broadcast(flag_tensor,
                                mpu.get_pipeline_model_parallel_last_rank(),
                                group=mpu.get_pipeline_model_parallel_group())
                best_score_save_flag = bool(flag_tensor.item())
        elif mpu.get_pipeline_model_parallel_world_size() > 1:
            flag_tensor = torch.cuda.FloatTensor([best_score_save_flag])
            torch.distributed.broadcast(flag_tensor,
                                        mpu.get_pipeline_model_parallel_last_rank(),
                                        group=mpu.get_pipeline_model_parallel_group())
            best_score_save_flag = bool(flag_tensor.item())


    if best_score_save_flag:
        optimizer = kwargs.get("optimizer", None)
        lr_scheduler = kwargs.get("lr_scheduler", None)

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            flag_tensor = torch.cuda.FloatTensor([new_iter_best_score])
            torch.distributed.broadcast(flag_tensor,
                            mpu.get_pipeline_model_parallel_last_rank(),
                            group=mpu.get_pipeline_model_parallel_group())
            new_iter_best_score = flag_tensor.item()

        print_rank_last(f"Basing key {args.eval_save_model_key} best score {new_iter_best_score} saving"
                        f" new checkpoint at iteration {iteration}...")

        if args.use_lora:
            convert_lora_to_linear_layer(model[0].module)
        save_checkpoint_and_time(iteration, model, optimizer,
                                    lr_scheduler, args.ckpt_nums)
        if args.use_lora:
            unfuse_lora_from_linear_layer(model[0].module)

        print_rank_last(f"Finish best score {new_iter_best_score} checkpoint saving.")
        return new_iter_best_score
    elif args.eval_save_model_key != None and best_score != None:
        return best_score
    else:
        return

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider, extra_models=None):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.reset_consumed_samples:
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
    else:
        if args.iteration > 0 and args.consumed_train_samples == 0:
            assert args.train_samples is None, \
                'only backward compatiblity support for iteration-based training'
            args.consumed_train_samples = args.iteration * args.global_batch_size
            update_num_microbatches(consumed_samples=args.consumed_train_samples)
            args.num_micro_batches = get_num_microbatches()
            args.current_global_batch_size = get_current_global_batch_size()
        if args.iteration > 0 and args.consumed_valid_samples == 0:
            # assert args.train_samples is None, \
            #     'only backward compatiblity support for iteration-based training'
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.valid_global_batch_size

    train_ds, valid_ds, test_ds = [None, None, None]
    # For mmp case, Data Loader is initialized on both encoder rank and TP 0 (although the one on TP 0 would not be used).
    # for no mmp case, Data loader only on TP 0.

    # if (not args.mmp_encoder_parallel and mpu.get_tensor_model_parallel_rank() == 0) or \
    #     (args.mmp_encoder_parallel and (mpu.is_mmp_encoder_stage() or mpu.get_tensor_model_parallel_rank() == 0)):
    # ========== every global rank read train data temporarily ========== 
    if True:
        # hack for zerocache+pp train_iters is None
        train_iters = None
        if args.train_iters is None:
            update_train_iters(args)
            train_iters = args.train_iters
            args.train_iters = None
        else:
            train_iters = args.train_iters
        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = train_iters * args.global_batch_size
        eval_iters = (train_iters // args.eval_interval + 1) * \
                     args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                      eval_iters * args.valid_global_batch_size,
                                      test_iters * args.valid_global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        if extra_models is not None:
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
                train_val_test_num_samples, extra_models)
        else:
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
                train_val_test_num_samples)

        if args.use_stream_dataset or args.use_iteration_dataset:
            do_train = train_ds is not None and train_iters > 0
            do_valid = valid_ds is not None and args.eval_iters > 0
            do_test = test_ds is not None and args.eval_iters > 0

        else:
            # Build dataloders.
            valid_dataloader_type = 'cyclic' if args.valid_dataloader_cyclic else args.dataloader_type
            if hasattr(args, 'build_pretraining_data_loader_func') and args.build_pretraining_data_loader_func is not None:
                train_dataloader = args.build_pretraining_data_loader_func(train_ds, args.consumed_train_samples)
                valid_dataloader = args.build_pretraining_data_loader_func(valid_ds, args.consumed_valid_samples)
                test_dataloader = args.build_pretraining_data_loader_func(test_ds, 0)
            else:
                train_dataloader = build_pretraining_data_loader(
                    train_ds, args.consumed_train_samples)
                valid_dataloader = build_pretraining_data_loader(
                    valid_ds, args.consumed_valid_samples, pin_memory=args.validation_pin_memory, dataloader_type=valid_dataloader_type)
                test_dataloader = build_pretraining_data_loader(test_ds, 0)

            # Flags to know if we need to do training/validation/testing.
            do_train = train_dataloader is not None and train_iters > 0
            do_valid = valid_dataloader is not None and args.eval_iters > 0
            do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    if args.use_multi_validation_dataset != None:
        for i in range(len(args.use_multi_validation_dataset)):
            this_dataset_name = f"{i}_{args.use_multi_validation_dataset[i].strip().split('/')[-1]}"
            args.multi_validation_dataset_name.append(this_dataset_name)
        args.multi_valid_dataset_num = len(args.multi_validation_dataset_name)

    else:
        args.multi_valid_dataset_num = 0

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']
    if args.use_stream_dataset or args.use_iteration_dataset:
        if mpu.get_tensor_model_parallel_rank() == 0:
            flags = torch.cuda.LongTensor([args.train_samples])
        else:
            flags = torch.cuda.LongTensor([0])
        torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
        args.train_samples = flags[0].item()
        args.train_iters = args.train_samples // args.global_batch_size
        print_rank_0('[use_stream_dataset] setting training iterations to {}'.format(args.train_iters))
        return  train_ds, valid_ds, test_ds

    if train_dataloader is not None:
        if not isinstance(train_dataloader, list):
            train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
                                else iter(cyclic_iter(train_dataloader))
        else:
            train_data_iterator = []
            for d_loader in train_dataloader:
                d_iter = iter(d_loader) if dl_type == 'single' \
                            else iter(cyclic_iter(d_loader))
                train_data_iterator.append(d_iter)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        if not isinstance(valid_dataloader, list):
            valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' and not args.valid_dataloader_cyclic \
                                else iter(cyclic_iter(valid_dataloader))
        else:
            valid_data_iterator = []
            for d_loader in valid_dataloader:
                d_iter = iter(d_loader) if dl_type == 'single' and not args.valid_dataloader_cyclic \
                            else iter(cyclic_iter(d_loader))
                valid_data_iterator.append(d_iter)
    else:
        if args.multi_valid_dataset_num > 0:
            valid_data_iterator = [None] * args.multi_valid_dataset_num
        else:
            valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
