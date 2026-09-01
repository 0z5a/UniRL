import inspect
import os
import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, Union, List, Optional

import torch
import torch.distributed as dist
from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex

from ..data_kits.samplers import RepeatRandomDistributedSampler


import deepspeed
from .helpers import (
    ScalarStates,
    CycleStates,
    save_checkpoint,
    get_trainable_params,
    WarmupCosineHelper
)
from ..constants import SCORE_METRICS, SAMPLE_METRICS, C_SCALE
from ..core.global_vars import get_nccl_timeout
from ..data_kits.csv_dataset import decode_csv_file
from ..ds_config import get_deepspeed_config
from ..models import build_model, EMA, DistributedEMA
from ..utils import lr_schedules
from ..utils.file_utils import (
    safe_dir,
    get_experiment_max_number,
    empty_logger,
    dump_configs,
    dump_codes,
    resolve_resume_path,
    logger_filter,
    dict_repr,
)
from ..utils.fsdp_wrapper import FSDPEngine
from ..utils.helpers import default, get_obj_from_str, to_2tuple
from ..utils.torch_utils import (
    set_manual_seed,
    set_reproducibility,
    build_optimizer,
    profiler_context,
    PRECISION_TO_TYPE,
    NAME_TO_SHARDING_STRATEGY,
)


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args

        self.dataset = None
        self.data_sampler = None
        self.dataloader = None

        self.resume_path = None

        self.sample_evaluator = None
        self.loss_evaluator = None
        self.score_evaluator = None

        self.init_env()
        # Keep the order: dataloader -> model & optimizer -> extra model -> evaluator
        self.build_dataloader()
        self.build_model_and_optimizer()
        self.build_extra_model()
        self.build_data_iterator()
        self.build_evaluator()

    def init_distributed_env(self):
        args = self.args
        if args.launcher == "deepspeed":
            deepspeed.init_distributed()
        else:
            dist.init_process_group(backend="nccl", timeout=get_nccl_timeout())

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        # Set current device for the current process, otherwise dist.barrier() will occupy more memory in rank 0.
        if args.launcher == "deepspeed":
            self.device = self.local_rank = args.local_rank
        else:
            self.device = self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.device)


        self.dp_rank = self.rank
        self.dp_size = self.world_size

    def init_env(self):
        args = self.args
        self.init_distributed_env()

        self.micro_batch_size = args.micro_batch_size
        self.grad_accu_steps = args.gradient_accumulation_steps
        self.global_batch_size = (
            args.global_batch_size
            if args.global_batch_size is not None
            else self.micro_batch_size * self.world_size * self.grad_accu_steps
        )

        # Setup seed for reproducibility or performance.
        set_manual_seed(args.global_seed)
        set_reproducibility(args.reproduce, args.global_seed, args.benchmark)

        self.output_dir = Path(args.output_dir)
        self.task_flag = args.task_flag
        if self.rank == 0:
            safe_dir(self.output_dir)
        dist.barrier()

        # Automatically increase the experiment number.
        existed_experiments = list(self.output_dir.glob("*"))
        experiment_index = get_experiment_max_number(existed_experiments) + 1
        model_name = args.model_name.replace("/", "").replace("-", "_")  # Replace '/' to avoid sub-directory.
        self.exp_dir = self.output_dir / f"{experiment_index:04d}_{model_name}_{self.task_flag}"
        self.ckpt_dir = self.exp_dir / "checkpoints"
        # Makesure all processes have the same experiment directory.
        dist.barrier()
        if self.rank == 0:
            safe_dir(self.exp_dir / "more_logs")
        dist.barrier()

        if self.rank == 0:
            safe_dir(self.ckpt_dir)
            from loguru import logger

            logger.add(
                self.exp_dir / "train.log",
                level="DEBUG",
                colorize=False,
                backtrace=True,
                diagnose=True,
                encoding="utf-8",
                filter=logger_filter("train"),
            )
            logger.add(
                self.exp_dir / "val.log",
                level="DEBUG",
                colorize=False,
                backtrace=True,
                diagnose=True,
                encoding="utf-8",
                filter=logger_filter("val"),
            )
            self.logger = logger.bind(name="train")
            self.val_logger = logger.bind(name="val")
        elif self.rank % 8 == 0:
            from loguru import logger

            logger.remove()
            logger.add(
                self.exp_dir / f"more_logs/train_{self.rank}.log",
                level="DEBUG",
                colorize=False,
                backtrace=True,
                diagnose=True,
                encoding="utf-8",
                filter=logger_filter("train"),
            )
            self.logger = logger.bind(name="train")
            self.val_logger = empty_logger()
        else:
            self.val_logger = self.logger = empty_logger()

        self.logger.info(f"Experiment directory created at: {self.exp_dir}")

        # Log and dump the configs and codes.
        self.logger.info(sys.argv)
        if self.rank == 0:
            # Dump the configs to a file.
            args_dict = dump_configs(args, self.exp_dir / "config.yaml")
            self.logger.info(dict_repr(args_dict))
            # Dump codes to the experiment directory.
            dump_codes(self.exp_dir / "codes.tar.gz",
                       root=Path(__file__).parents[2],
                       sub_dirs=["hymm", "processors", "jobs"],
                       save_prefix=self.task_flag,
                       )

    def resume_dataloader(self, ss):
        # Move sampler to start_index
        if isinstance(self.data_sampler, BlockDistributedSampler):
            start_index = ss.epoch_consumed_samples_per_dp
        elif isinstance(self.data_sampler, DistributedSamplerWithStartIndex):
            start_index = ss.epoch_consumed_samples_total
        else:
            raise NotImplementedError(
                "Only BlockDistributedSampler and DistributedSamplerWithStartIndex support --resume-dataloader. "
                f"Got {type(self.data_sampler)}."
            )
        # Move sampler to start_index
        self.data_sampler.start_index = start_index

    def get_states_cls(self, state_type):
        if state_type == 'scalar':
            return ScalarStates
        elif state_type == 'cycle':
            return CycleStates
        else:
            raise ValueError(f"Unknown state type: {state_type}")

    def get_trainable_params(self, model, training_parts):
        self.trainable_params = get_trainable_params(model, training_parts)
        if isinstance(self.trainable_params, list):
            self.num_trainable_params = sum(
                p.numel() if isinstance(p, torch.Tensor)
                else sum(p2.numel() for p2 in p['params'])
                for p in self.trainable_params
            )
        else:
            self.num_trainable_params = -1
        return self.trainable_params

    def model_init_post_hook(self):
        pass
    def model_init_pre_hook(self):
        pass

    def apply_fsdp(self, deepspeed_config, lr_scheduler, process_group=None):
        # Initialize FSDP
        self.logger.info("Using FSDP")
        args = self.args
        self.get_trainable_params(self.model, args.training_parts),
        mp_policy = MixedPrecision(
            param_dtype=PRECISION_TO_TYPE[args.param_dtype],
            reduce_dtype=PRECISION_TO_TYPE[args.reduce_dtype],
            buffer_dtype=PRECISION_TO_TYPE[args.buffer_dtype],
        )
        wrap_module = get_obj_from_str(args.wrap_module)
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                wrap_module,
            },
        )
        optimizer_cls = build_optimizer(args)
        self.model_engine = FSDPEngine(
            args,
            deepspeed_config,
            self.model,
            optimizer_cls,
            lr_scheduler,
            logger=self.logger,
            process_group=process_group,
            mixed_precision=mp_policy,
            sharding_strategy=NAME_TO_SHARDING_STRATEGY[args.sharding_strategy],
            device_id=torch.cuda.current_device(),
            auto_wrap_policy=wrap_policy,
            use_orig_params=True,
            sync_module_states=True,
        )
        self.optimizer = self.model_engine.optimizer
        self.lr_scheduler = self.model_engine.lr_scheduler

    def build_model(self, dtype=None, device=None):
        return build_model(self.args, logger=self.logger, dtype=dtype, device=device)

    def initialize_puretorch_model_engine(self):
        raise NotImplementedError

    def build_model_and_optimizer(self, process_group=None):
        args = self.args
        self.logger.info("Building model and optimizer...")

        self.model_init_pre_hook()
        self.model, self.model_settings = self.build_model(dtype=PRECISION_TO_TYPE[args.precision], device=self.device)
        if self.rank == 0:
            # Dump the model settings to a file.
            model_settings = dump_configs(self.model_settings, self.exp_dir / "model_settings.yaml")
            self.logger.info(dict_repr(model_settings))

        if args.reproduce:
            self.model.enable_deterministic()

        # After model initialization, we set different seed for each process.
        set_manual_seed(args.global_seed + self.dp_rank)

        # =========================== Load model weights ==========================
        # Make sure resume training states before initializing deepspeed.
        resume_path = None
        resume_tag = None
        need_model_engine_resume = False    # resume model_engine (by deepspeed)
        scalar_state = {
            'lr': args.lr,
        }                                   # resume ScalarState
        ema_resume_state_dict = None        # resume EMA
        ema_resume_decay_steps = None       # resume EMA configs
        client_state = {}
        if args.resume:
            resume_path = resolve_resume_path(args.resume, self.output_dir)
            if resume_path.is_file():
                # If resume_path is a file, it must be the *_model_state.pt file. In this case, we believe the user don't
                # want to resume the optimizer states.
                if args.launcher == "deepspeed":
                    assert resume_path.name.endswith(
                        "_model_states.pt"
                    ), f"A model state file should end with '_model_states.pt', but got {resume_path.name}."
                self.resume_path = resume_path
                self.logger.info(f"Loading model states from {resume_path}...")
                # client_state = torch.load(resume_path, map_location=lambda storage, loc: storage)
                client_state = torch.load(resume_path, map_location='cpu', mmap=True)
                self.logger.info("Loaded.")
                # Make sure use strict=True to avoid unexpected errors.
                self.model.load_state_dict(client_state["module"], strict=True)
                self.logger.info("Model states loaded.")

            elif resume_path.is_dir():
                # If resume_path is a directory, it must be the checkpoint directory or the <steps> directory.
                # In this case, we believe the user want to resume both the model states and optimizer states.
                # We will load the optimizer states after initializing deepspeed.
                if (resume_path / "checkpoints").exists():
                    resume_path = resume_path / "checkpoints"
                self.resume_path = resume_path
                if not (resume_path / "latest").exists():
                    resume_tag = resume_path.name
                    resume_path = resume_path.parent
                need_model_engine_resume = True
            else:
                raise ValueError(f"Unknown resume path: {resume_path}")

        # ========================== Initialize model_engine, optimizer =========================
        # Note: Model/Pipeline parallel is not supported yet. So, data-parallel-size equals to world-size.
        desired_gbs = self.micro_batch_size * self.world_size * self.grad_accu_steps
        if self.global_batch_size is not None:
            assert self.global_batch_size == desired_gbs, (
                f"Global batch size should be {desired_gbs}={self.micro_batch_size}*{self.world_size}*{self.grad_accu_steps}, "
                f"but got {self.global_batch_size}."
            )
        else:
            self.global_batch_size = self.micro_batch_size * self.world_size * self.grad_accu_steps

        deepspeed_config = get_deepspeed_config(args)
        if args.tensorboard:
            deepspeed_config["tensorboard"] = {
                "enabled": True,
                "output_path": str(self.output_dir.absolute()),
                "job_name": self.exp_dir.name,
            }

        # Build the learning rate scheduler.
        lr_scheduler = getattr(lr_schedules, default(args.lr_schedule, getattr(args, 'lr_scheduler_name', None)))
        lr_scheduler_params = dict(**args.get('lr_scheduler_params', {}))
        if "total_num_steps" in set(inspect.signature(lr_scheduler).parameters.keys()) and \
                lr_scheduler_params.get("total_num_steps", -1) < 0:
            # Calculate the total number of steps based on the number of epochs.
            try:
                iters_per_epoch = len(self.dataloader) // self.grad_accu_steps
            except NotImplementedError:
                raise ValueError(f"{args.lr_schedule} requires `total_num_steps`, but length of dataloader is undefined."
                                 f"Please explicitly set `total_num_steps` in `lr_scheduler_params`.")
            max_iter = args.max_epochs * iters_per_epoch
            lr_scheduler_params["total_num_steps"] = max_iter
            self.logger.info(f"Learning rate scheduler: set total_num_steps to {max_iter} based on the number of epochs.")
        lr_scheduler = partial(
            lr_scheduler,
            **lr_scheduler_params,
        )

        self.model_init_post_hook()
        if args.launcher == "deepspeed":
            # Initialize deepspeed.
            self.logger.info("Using deepspeed")
            self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                args=self.args,
                config_params=deepspeed_config,
                model=self.model,
                model_parameters=self.get_trainable_params(self.model, args.training_parts),
                lr_scheduler=lr_scheduler,
            )
        elif args.launcher == "pure_torch":
            self.get_trainable_params(self.model, args.training_parts)
            self.initialize_puretorch_model_engine()
        else:
            raise DeprecationWarning('launcher should be deepspeed or pure_torch. fsdp launcher is deprecated')
            self.apply_fsdp(deepspeed_config, lr_scheduler, process_group)

        if need_model_engine_resume:
            # Resume model_states [and optim_states].
            load_path, client_state = self.model_engine.load_checkpoint(
                resume_path, resume_tag, load_optimizer_states=not args.no_load_optim_states)
            if load_path is None:
                raise ValueError(f"Failed to load checkpoint from {resume_path}.")

        if args.resume:
            # Resume EMA
            if args.use_ema and not args.no_load_ema:
                if args.distributed_ema:
                    if resume_path.is_file():
                        dist_ema_path = resume_path.parent / "dist_ema"
                    elif resume_path.is_dir():
                        if resume_path.name == "checkpoints":
                            with open(resume_path / "latest", "r") as f:
                                resume_tag = f.read()
                                dist_ema_path = resume_path / resume_tag / "dist_ema"
                        else:
                            dist_ema_path = resume_path / "dist_ema"
                    else:
                        raise ValueError(f"Unknown resume path: {resume_path}")
                    
                    try:
                        saved_ema = torch.load(dist_ema_path / f"{self.rank}.pt", map_location='cpu', mmap=True)
                        ema_resume_state_dict = saved_ema["ema"]
                        ema_resume_decay_steps = saved_ema["ema_config"]["decay_steps"]
                    except Exception as e:
                        self.logger.error(f"Failed to resume Distributed EMA from {dist_ema_path}. {type(e)}: {e}")
                        ema_resume_state_dict = None
                        ema_resume_decay_steps = None
                else:
                    # sometimes we resume from a model which doesn't have ema, but we want to use ema after resume
                    ema_resume_state_dict = client_state["ema"] if "ema" in client_state else None
                    ema_resume_decay_steps = (
                        client_state["ema_config"]["decay_steps"] if "ema_config" in client_state else None
                    )

            # Resume ScalarStates.
            self.ss = self.get_states_cls('scalar').from_pretrained(
                client_state["scalar_state"],
                rank=self.rank,
                world_size=self.world_size,
                default_rank0_ss=args.default_rank0_ss,
                default=scalar_state,
            )

            # Fix LR Scheduler. Depend on optim_states.
            if args.fix_lr_scheduler:
                # Only support WarmupCosineLR now.
                assert args.lr_schedule == "WarmupCosineLR", f"Only support WarmupCosineLR now, got {args.lr_schedule}."
                self.lr_helper = WarmupCosineHelper(lr=args.lr,
                                                    warmup_num_steps=lr_scheduler_params.get('warmup_num_steps', 1000),
                                                    cos_min_ratio=lr_scheduler_params.get('cos_min_ratio', 0.0001),
                                                    start_lr=self.ss.lr,
                                                    start_step=self.ss.update_steps,
                                                    total_num_steps=lr_scheduler_params['total_num_steps'])
            else:
                self.lr_helper = lambda x: x
            # Resume LR Scheduler.
            self.lr_scheduler.step(last_batch_iteration=self.lr_helper(self.ss.update_steps))

            # Resume dataloader.
            if args.resume_dataloader:
                self.resume_dataloader(self.ss)
            else:
                # If don't resume dataloader, we manually reset the epoch_* states in self.ss
                self.ss.reset_epoch_based_states()

        elif self.args.get("resume_puretorch", False):
            # resume ckpt by Engine
            pass
        
        else:
            self.ss = self.get_states_cls('scalar')(**scalar_state)
            self.lr_helper = lambda x: x

        if resume_path:
            self.logger.info(f"Model states loaded from {resume_path}.")

        self.logger.info(f"Training states initialized: \n{self.ss}")

        # Mixed precision training.
        self.target_dtype = PRECISION_TO_TYPE[args.autocast_dtype]
        self.autocast_enabled = self.target_dtype != torch.float32

        # ============================= Build EMA model ===========================
        self.ema = None
        if args.use_ema:
            self.logger.info("Building EMA model...")
            ema_kwargs = dict(dtype=args.ema_precision,
                              decay=args.ema_decay,
                              warmup=args.ema_warmup,
                              power=args.ema_warmup_power,
                              logger=self.logger)
            if args.distributed_ema:
                self.ema = DistributedEMA(self.world_size, self.rank, self.model, **ema_kwargs)
            else:
                self.ema = EMA(self.model, **ema_kwargs)
            if args.resume and ema_resume_state_dict is not None:
                self.ema.load_state_dict(ema_resume_state_dict)
                self.ema.set_decay_steps(ema_resume_decay_steps)

    def build_extra_model(self):
        """leave tokenizer to be implemented by subclass,
        since we may need to do experiments on different modalities with different tokenizers
        """
        raise NotImplementedError()

    def build_dataloader(self):
        raise NotImplementedError()

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        raise NotImplementedError()

    def build_data_iterator(self):
        pass

    def before_train(self):
        args = self.args

        # ============================= Print key info =============================
        print(f"[{self.rank}] Worker ready.")
        dist.barrier()

        try:
            iters_per_epoch = len(self.dataloader) // self.grad_accu_steps
        except NotImplementedError:
            iters_per_epoch = 0
        if hasattr(self.model, "params_count"):
            self.params_count = self.model.params_count()
        else:
            self.params_count = {
                "total": sum(p.numel() for p in self.model.parameters()),
                "attn+mlp": sum(p.numel() for name, p in self.model.named_parameters() if "attn" in name or "mlp" in name),                
            }
        self.logger.info("****************************** Running training ******************************")
        self.logger.info(f"  Number GPUs:               {self.world_size}")
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.logger.info(f"  Training samples(total):   {len(self.dataset):,}({self.dataset.total_length:,})")
        elif hasattr(self, 'dataset_dict'):
            for k, v in self.dataset_dict.items():
                self.logger.info(f"  Training samples:          {k} = {len(v):,}({v.total_length:,})")
        for k, v in self.params_count.items():
            self.logger.info(f"  Number {k} parameters:   {v:,}")
        self.logger.info(f"  Number trainable params:   {self.num_trainable_params:,}")
        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info(f"  Updates per epoch:         {iters_per_epoch:,}" + ("(unknown)" if iters_per_epoch == 0 else ""))
        self.logger.info(f"  Batch size per device:     {self.micro_batch_size}")
        self.logger.info(f"  Batch size all device:     {self.global_batch_size}")
        self.logger.info(f"  Gradient Accu steps:       {self.grad_accu_steps}")
        self.logger.info(f"  Training epochs:           {self.ss.epoch}/{args.max_epochs}")
        self.logger.info(f"  Training total steps:      {self.ss.update_steps:,}/{args.max_training_steps:,}")
        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info(f"  Main model precision:      {args.precision}")
        self.logger.info(f"  Autocast precision:      {args.autocast_dtype}")
        self.logger.info(f"  Using EMA model:           {args.use_ema}")
        if args.use_ema:
            self.logger.info(f"      Using Distributed EMA: {args.distributed_ema}")
            self.logger.info(f"      EMA precision:         {args.ema_precision}")
            self.logger.info(f"      EMA decay:             {self.ema.decay if args.use_ema else None}")
            self.logger.info(f"      EMA warmup power:      {self.ema.power if args.use_ema else None}")
        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info(f"  Media Tokenizer:           {args.vae_type} ({args.vae_precision})")
        self.logger.info(f"  VAE autocast precision:           {args.vae_autocast_dtype}")
        if hasattr(self, 'vae') and hasattr(self.vae, 'codebook_size'):
            self.logger.info(f"      Codebook size:         {self.vae.codebook_size}")
        if hasattr(self, 'vae') and hasattr(self.vae, 'downsample_factor'):
            self.logger.info(f"      Downsample factor:     {self.vae.downsample_factor}")
        self.logger.info("------------------------------------------------------------------------------")
        if self.resume_path:
            self.logger.info(f"  Resume from:               {self.resume_path}")
        self.logger.info(f"  Experiment directory:      {self.exp_dir}")
        self.logger.info("*******************************************************************************")

    def after_train(self):
        self.logger.info("Training Finished!")

    def train_step(self, batch):
        start1 = time.time()
        model_input_kwargs, cur_batch_size, n_tokens = self.prepare_model_inputs(batch, self.device)
        torch.cuda.synchronize()
        duration1 = time.time() - start1

        start2 = time.time()
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            loss_dict = self.model_engine(**model_input_kwargs)
        torch.cuda.synchronize()
        duration2 = time.time() - start2

        times = {
            "preprocess": duration1,
            "forward": duration2,
        }

        return loss_dict, cur_batch_size, n_tokens, times

    def shuffle_dataset_and_set_start_index(self, ss):
        # Makesure all processors use the same seed to shuffle dataset.
        if self.args.shuffle_by == "sampler":
            self.logger.info(f"Shuffle by sampler with epoch={ss.epoch}")
            self.dataloader.sampler.set_epoch(ss.epoch)
        else:
            self.logger.info(f"Shuffle by dataset with seed={self.args.global_seed + ss.epoch}, fast={self.args.fast_shuffle}")
            self.dataset.shuffle(self.args.global_seed + ss.epoch, fast=self.args.fast_shuffle)
        self.logger.info(f"End of random shuffle")
        # Set start index
        if hasattr(self.dataloader.sampler, "start_index"):
            if isinstance(self.dataloader.sampler, BlockDistributedSampler):
                self.dataloader.sampler.start_index = ss.epoch_consumed_samples_per_dp
            elif isinstance(self.dataloader.sampler, DistributedSamplerWithStartIndex) or isinstance(self.dataloader.sampler, RepeatRandomDistributedSampler):
                self.dataloader.sampler.start_index = ss.epoch_consumed_samples_total
            else:
                raise NotImplementedError(
                    "Only BlockDistributedSampler and DistributedSamplerWithStartIndex support --resume-dataloader. "
                    f"Got {type(self.dataloader.sampler)}."
                )

    def update_train_states(self, ss, cs, batch, batch_size, n_tokens, loss):
        # A forward-backward step is counted as one train step.
        ss.add(
            train_steps=1,
            epoch_train_steps=1,
            epoch_consumed_samples_per_dp=batch_size,
        )
        cs.add(
            log_steps=1,
            running_loss=loss,
            running_samples=batch_size,
            running_tokens=batch_size * n_tokens,
        )
        # We enable `is_update_step` if the current step is the gradient accumulation boundary.
        is_update_step = self.ss.train_steps % self.grad_accu_steps == 0
        # TODO(kevinkhwu): 似乎现在 grad_accu 是在 内部判断更新，而不是 hy_parallelism 判断
        #     考虑统一一下
        if is_update_step:
            # A ([forward-backward] x grad_accu)-update step is counted as one update step.
            ss.add(
                update_steps=1,
                epoch_update_steps=1,
                current_run_update_steps=1
            )
            ss.lr = self.optimizer.param_groups[0]["lr"]

        return is_update_step

    def update_log_states(self, ss, all_cs):
        cum_samples = sum([cs_i.running_samples for cs_i in all_cs])
        cum_tokens = sum([cs_i.running_tokens for cs_i in all_cs])
        ss.add(
            epoch_consumed_samples_total=cum_samples,
            consumed_samples_total=cum_samples,
            consumed_tokens_total=cum_tokens,
            consumed_computations_attn=6 * self.params_count["attn+mlp"] * cum_tokens / C_SCALE,
            consumed_computations_total=6 * self.params_count["total"] * cum_tokens / C_SCALE,
        )
        return cum_samples

    def get_events(self, ss, loss):
        log_events = [
            f"Consumed Samples: {ss.consumed_samples_total:,}",
            f"Consumed Tokens: {ss.consumed_tokens_total:,}",
        ]
        summary_events = [
            ("Train/Tokens/train_loss", loss, ss.consumed_tokens_total),
        ]
        return log_events, summary_events

    def train_loop(self):
        args = self.args
        self.model_engine.train()
        self.ss.current_run_update_steps = 0

        if args.init_save:
            save_checkpoint(args, self.rank, self.logger, self.model_engine, self.ema, self.ss, self.ckpt_dir)

        # Training loop
        start_epoch = self.ss.epoch
        finished = False
        nan_grad_count = 0

        for epoch in range(start_epoch, args.max_epochs):
            self.shuffle_dataset_and_set_start_index(self.ss)

            with profiler_context(
                args.profile, self.exp_dir, worker_name=f"Rank_{self.rank}"
            ) as prof:
                self.logger.info(f"Beginning epoch {epoch}...")
                try:
                    self.logger.info(f"  Steps left this epoch: {len(self.dataloader) // self.grad_accu_steps:,}")
                except NotImplementedError:
                    pass
                # Define cycle states, which accumulate the training information between log_steps.
                cs = self.get_states_cls('cycle')()
                torch.cuda.synchronize()
                start_time = time.time()
                data_start = time.time()
                times = {}

                for bi, batch in enumerate(self.dataloader):
                    torch.cuda.synchronize()
                    times['data'] = time.time() - data_start

                    # Dry run dataloader to check data processing.
                    if args.get('dry_run_dataloader'):
                        if bi > 0 and bi % 20 == 0:
                            self.logger.info(
                                f"Dry run dataloader: {bi} batches processed. Average time: {times['data'] / 20:.2f}s."
                            )
                            data_start = time.time()

                        continue

                    loss_dict, batch_size, n_tokens, forward_times = self.train_step(batch)
                    times.update(forward_times)

                    backward_start = time.time()
                    loss = loss_dict["loss"].mean()
                    for k, v in loss_dict.items():
                        if "loss" in k and k != "loss":
                            cs.running_sub_loss_dict[k] += v.mean().item()
                            cs.running_sub_step_dict[k] += 1
                    self.model_engine.backward(loss)
                    torch.cuda.synchronize()
                    times['backward'] = time.time() - backward_start

                    is_update_step = self.update_train_states(self.ss, cs, batch, batch_size, n_tokens, loss.item())

                    if args.skip_nan_grad and hasattr(self.model_engine.optimizer, "scaled_global_norm"):
                        scaled_grad_norm = self.model_engine.optimizer.scaled_global_norm()     
                        if torch.any(torch.isnan(scaled_grad_norm)):
                            nan_grad_count += 1
                            self.logger.info(f"Step {self.ss.update_steps:07d} grad norm is nan, skipping step. Total nan grad count: {nan_grad_count}.")
                            self.model_engine.optimizer.zero_grad()

                    # Update model parameters at the boundary of gradient accumulation.
                    update_start = time.time()
                    # Get the lr before optimizer.step()
                    lrs = [group["lr"] for group in self.optimizer.param_groups]
                    self.model_engine.step(lr_kwargs={"last_batch_iteration": self.lr_helper(self.ss.update_steps)})
                    torch.cuda.synchronize()
                    times['update'] = time.time() - update_start

                    if self.ss.update_steps >= args.max_training_steps:
                        # Enter stopping routine if max steps reached after this step.
                        finished = True

                    # Update EMA model at the step of main model parameters update.
                    if args.use_ema and is_update_step:
                        self.ema.update(self.model_engine.module)

                    # Log training information:
                    if is_update_step and self.ss.update_steps % args.log_every == 0:
                        # All-gather scalar states and cycle states.
                        all_cs: List[Optional[CycleStates]] = [None for _ in range(self.world_size)]
                        torch.distributed.all_gather_object(all_cs, cs)

                        # Calculate average main loss
                        avg_loss = sum([cs_i.running_loss for cs_i in all_cs]) / sum([cs_i.log_steps for cs_i in all_cs])
                        # Calculate average sub losses.
                        merged_loss_dict = {}
                        merged_step_dict = {}
                        for cs_i in all_cs:
                            for k, v in cs_i.running_sub_loss_dict.items():
                                if k not in merged_loss_dict:
                                    merged_loss_dict[k] = v
                                    merged_step_dict[k] = cs_i.running_sub_step_dict[k]
                                else:
                                    merged_loss_dict[k] += v
                                    merged_step_dict[k] += cs_i.running_sub_step_dict[k]
                        sorted_keys = sorted(list(merged_loss_dict.keys()))
                        avg_sub_loss_dict = {k: merged_loss_dict[k] / merged_step_dict[k] for k in sorted_keys}
                        # Calculate cumulated metrics.
                        cum_samples = self.update_log_states(self.ss, all_cs)

                        # Synchronize cuda to accurately measure training speed:
                        torch.cuda.synchronize()
                        end_time = time.time()
                        steps_per_sec = cs.log_steps / self.grad_accu_steps / (end_time - start_time)
                        seconds_per_step = (end_time - start_time) / (cs.log_steps / self.grad_accu_steps)
                        samples_per_sec = cum_samples / (end_time - start_time)

                        grad_norm = self.model_engine.get_global_grad_norm()
                        user_log_events, user_summary_events = self.get_events(self.ss, avg_loss)

                        log_events = [
                             f"Train Loss: {avg_loss:.4f}",
                             *[f"{k}: {v:.4f}" for k, v in avg_sub_loss_dict.items()],
                         ] + [f"Lr{lr_i}: {lr:.6g}" for lr_i, lr in enumerate(lrs)] + [
                             f"Steps/Sec: {steps_per_sec:.2f}",
                             f"Sec/Step: {seconds_per_step:.2f}",
                             f"Samples/Sec: {int(samples_per_sec):d}",
                             f"Global Grad Norm: {grad_norm:.4f}",
                             f"Nan Grad Count: {nan_grad_count}",
                         ] + user_log_events + [
                             f"T{time_key}: {duration:.4f}"
                             for time_key, duration in times.items()
                         ]
                        summary_events = [
                            ("Train/Steps/train_loss", avg_loss, self.ss.update_steps),
                            *[("Train/Steps/" + k, v, self.ss.update_steps) for k, v in avg_sub_loss_dict.items()],
                            ("Train/Steps/LR", self.ss.lr, self.ss.update_steps),
                            ("Train/Steps/steps_per_sec", steps_per_sec, self.ss.update_steps),
                            ("Train/Steps/samples_per_sec", int(samples_per_sec), self.ss.update_steps),
                            ("Train/Steps/seconds_per_step", seconds_per_step, self.ss.update_steps),
                            ("Train/Steps/grad_norm", grad_norm, self.ss.update_steps),
                            ("Train/ComputationsAttn/train_loss", avg_loss, self.ss.consumed_computations_attn),
                            ("Train/ComputationsTotal/train_loss", avg_loss, self.ss.consumed_computations_total),
                        ] + user_summary_events
                        # Log the training information to the logger.
                        self.logger.info(f"(step={self.ss.update_steps:07d}) " + ", ".join(log_events))
                        # Log the training information to the monitor.
                        if self.model_engine.monitor.enabled and self.rank == 0:
                            self.model_engine.monitor.write_events(summary_events)

                        # Reset monitoring variables:
                        cs.reset()
                        start_time = time.time()

                    # Save checkpoint:
                    if (is_update_step and self.ss.update_steps % args.ckpt_every == 0) or (
                        finished and args.final_save
                    ):
                        self.save_checkpoint()

                    # Perform evaluation
                    if args.validation_every > 0 and (
                        (is_update_step and self.ss.update_steps % args.validation_every == 0)
                        or (
                            is_update_step
                            and self.ss.current_run_update_steps in args.validation_at_steps
                        )
                        or finished
                    ):
                        # Clear the cache to save GPU memory.
                        torch.cuda.empty_cache()
                        # del loss_dict
                        self.model_engine.module.eval()
                        self.val_logger.info(
                            f"Start evaluation after train epoch={self.ss.epoch}, step={self.ss.update_steps} "
                            + (f"(update_step={self.ss.update_steps:07d}) " if self.grad_accu_steps > 1 else "")
                        )
                        with torch.no_grad():
                            self.eval_step(loss_dict)
                        # Wait for rank 0 finished processing and saving
                        dist.barrier()
                        # Return to training mode
                        self.model_engine.module.train()
                        # Clear the cache to save GPU memory.
                        torch.cuda.empty_cache()

                    if prof:
                        prof.step()

                    if finished:
                        self.logger.info(f"Finished and breaking loop at step={self.ss.update_steps}.")
                        break

                    torch.cuda.synchronize()
                    data_start = time.time()

                if finished:
                    self.logger.info(f"Finished and breaking loop at epoch={epoch}.")
                    break

                # Reset epoch states
                new_epoch = self.ss.inc_epoch()
                self.logger.info(f"Increase epoch to {new_epoch}.")

    def save_checkpoint(self):
        save_checkpoint(self.args, self.rank, self.logger, self.model_engine, self.ema, self.ss, self.ckpt_dir)

    def train(self):
        self.before_train()
        self.train_loop()
        self.after_train()

    def get_sampler(self):
        """get sampler
        """
        raise NotImplementedError()

    def build_evaluator(self):
        args = self.args

        val_metrics = default(args.validation_metrics, [])
        # (not []) is True
        if not val_metrics:
            return

        metric_names = [metric.split('@')[0] for metric in val_metrics]

        sample_metrics_simple = set(metric_names) & SAMPLE_METRICS
        if sample_metrics_simple:
            self.sample_evaluator = self.get_sampler()
            self.logger.info(f"Enable sample evaluator: {self.sample_evaluator.__class__.__name__}")
            self.sample_metrics = sample_metrics_simple
            self.logger.info(f"  Metrics: {self.sample_metrics}")
            # Try to decode csv file to check if it is valid.
            _ = decode_csv_file(
                args.csv, error_message="When using `sample` metrics, you must specify a valid CSV file by --csv.")

        score_metrics_simple = set(metric_names) & SCORE_METRICS
        if score_metrics_simple:
            self.score_evaluator = self.sample_evaluator or self.get_sampler()
            self.logger.info(f"Enable score evaluator: {self.score_evaluator.__class__.__name__}")
            self.score_metrics = sorted([
                metric for metric in val_metrics if metric.split('@')[0] in score_metrics_simple
            ])
            self.logger.info(f"  Metrics: {self.score_metrics}")

    # this function is implemented for image tasks, you should override this function on request
    def eval_step(self, results_dict=None):
        args = self.args
        # Performing validation on multiple image sizes.
        val_summary_events = []

        if self.sample_evaluator is not None:
            self.val_logger.info(f"---------------------------- Sample Evaluator --------------------------")
            image_size = to_2tuple(args.sample_image_size)
            suffix = self.sample_evaluator.get_sample_dir_suffix(
                testset=Path(args.csv).stem,
                image_size=image_size,
                load_key="module",
            )
            save_template = str(self.exp_dir / "samples" / f"{self.ss.update_steps:07d}_{suffix}" / "{}_{{}}.png")
            dataloader = self.sample_evaluator.build_sample_dataloader(
                data_source=args.csv,
                save_template=save_template,
                batch_size=args.sample_batch_size,
                seed_type=args.seed_type,
                seed=args.seed,
            )
            self.sample_evaluator.batch_sample(
                dataloader=dataloader,
                size=image_size,
            )

        if self.score_evaluator is not None:
            self.val_logger.info(f"---------------------------- Score Evaluator ----------------------------")
            image_size = to_2tuple(args.metric_image_size)
            size = f"{image_size[0]}x{image_size[1]}"
            save_template = str(self.exp_dir / "evaluation" / f"{self.ss.update_steps:07d}_score_{size}.json")

            self.score_evaluator.initialize_scores(self.score_metrics)
            self.val_logger.info(f"Loading score models...")
            self.score_evaluator.load_score_models(min(image_size))
            image_save_base = (
                self.score_evaluator.get_sample_save_dir(testset=None, image_size=image_size)
                if args.validation_metrics_save_image
                else None
            )
            results = self.score_evaluator.eval(
                image_size=image_size,
                batch_size=args.metric_batch_size,
                save_path=save_template,
                image_save_base=image_save_base,
                extra_save_info=self.score_evaluator.get_infer_kwargs(),
            )
            # Release score models to save GPU memory.
            self.val_logger.info(f"  Releasing score models...")
            self.score_evaluator.release_score_models(min(image_size))
            for item in results:
                val_summary_events.append(
                    (f"Val/Steps/{item['metric']}_{size}", item["value"], self.ss.update_steps)
                )

        if self.model_engine.monitor.enabled and self.rank == 0 and val_summary_events:
            self.model_engine.monitor.write_events(val_summary_events)
