# This trainer is based on pure torch.
# This trainer uses the same interface as PTMv2 trainer for ease of integration.
#
import addict
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Type, Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.optim import AdamW
from hy_parallelism.parallel_states import init_parallel_state
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from ..core.data_provider import (
    DatasetsProvider,
    prepare_model_inputs,
)
from ..core.extra_model_provider import (
    build_scalar_state,
    build_vae,
    build_denoiser,
    build_tkwrapper,
)
from ..core.global_vars import (
    get_combined_iterator,
    get_mm_state,
    get_nccl_timeout,
    set_args,
    set_logger,
)
from ..core.parallel_states import ParallelState
from ..engines import find_engine
from ..engines.hy_base_engine import HyBaseEngine
from ..models import build_model
from ..models.multimodal.hunyuan_multimodal_state import ParameterSummary
from ..utils.env import is_bitwise_align_mode
from ..utils.file_utils import empty_logger, dump_configs, dump_codes
from ..utils.helpers import print_args
from ..utils.torch_utils import set_manual_seed, set_reproducibility, Timer, profiler_context
from ..utils.moe_utils import update_router_expert_bias
from ..utils.optimzer_utils import build_optimizer_factory_from_model_args, pre_optimizer_hook

if TYPE_CHECKING:
    from ..data_kits.combined_iterator import CombinedBatchIterator
    from ..data_kits.utils import MultimodalTasksState
    from ..trainers.helpers import MultiModalScalarStates


class MultimodalTrainer(object):
    p_state: ParallelState
    mm_state: "MultimodalTasksState"
    scalar_state: "MultiModalScalarStates"
    combined_iterator: "CombinedBatchIterator"

    def __init__(self, args: Namespace):
        self.args: Namespace = args

        set_args(args)
        self.args = args
        self.timer = Timer(enabled=args.use_timer)

        self.init_env()
        self.build_extra_model()
        self.build_model_and_optimizer()
        self.build_dataloader()
        self.after_initialize()

    def init_env(self):
        args = self.args
        self.init_distributed_env()

        self.micro_batch_size = args.micro_batch_size
        self.grad_accu_steps = args.gradient_accumulation_steps
        self.global_batch_size = args.global_batch_size

        # Setup seed for reproducibility or performance.
        set_manual_seed(args.seed)
        set_reproducibility(args.reproduce, args.seed, args.benchmark)

        self.logger.info(f"Reproduce: {args.reproduce}, Global Seed: {args.seed}, Benchmark: {args.benchmark}")
        self.checkpoint_dir = Path(args.save)
        self.exp_dir = self.checkpoint_dir.parent
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.task_id = args.task_id
        self.logger.info(f"Task ID set to: {self.task_id}")
        self.test_loss_dump_file = getattr(args, "test_loss_dump_file", None)
        self.test_loss_values: list[float] = []
        if self.test_loss_dump_file:
            self.logger.info(
                f"Test loss dump enabled. output={self.test_loss_dump_file}"
            )

        # Dump the configs and codes.
        if self.rank == 0:
            print_args("arguments", args)
            # Dump the configs to a file.
            dump_configs(args, self.exp_dir / f"runtime_configs/{self.task_id}_args.yaml")
            # Dump codes to the experiment directory.
            dump_codes(self.exp_dir / f"codes/{self.task_id}_codes.tar.gz",
                       root=Path(__file__).parents[2],
                       sub_dirs=["hymm", "processors", "jobs"],
                       valid_suffixes=[".py", ".sh"],
                       save_prefix=self.task_id,
                       )

        # Wait for rank 0 dumping configs and codes to finish initialization.
        dist.barrier()

    def init_distributed_env(self):
        args = self.args
        nccl_timeout = get_nccl_timeout()
        dist.init_process_group(backend="nccl", timeout=nccl_timeout)
        self.device = args.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        # Set current device for the current process, otherwise dist.barrier() will occupy more memory in rank 0.
        torch.cuda.set_device(self.device)

        if self.rank == 0:
            from loguru import logger
            self.logger = logger
        else:
            self.logger = empty_logger()
        set_logger(self.logger)

        # Check and formalize parallel settings
        for parallel_name in ["pipeline_model_parallel_size", "tensor_model_parallel_size"]:
            if getattr(args, parallel_name) > 1:
                raise ValueError(f"{parallel_name} is not supported in MultimodalTrainer yet.")
        if (ep_size := getattr(args, "expert_model_parallel_size", 1)) > 1 \
                and getattr(args, "moe_impl", None) != "ep_moe":
            raise ValueError(f"When moe_impl is not 'ep_moe', expert_model_parallel_size({ep_size}) should be set to 1.")

        dp_shard = min(
            args.dp_shard if args.dp_shard > 0 else 8,
            self.world_size,
        )
        dp_replicate = args.dp_replicate if args.dp_replicate > 0 else -1
        init_parallel_state(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            # advanced parallels
            tp=args.tensor_model_parallel_size,
            pp=args.pipeline_model_parallel_size,
            ep=args.expert_model_parallel_size,
            cp=args.context_parallel_size,
            timeout=nccl_timeout,
        )
        self.p_state = ParallelState.from_pure_torch()
        self.dp_rank = self.p_state.dp_rank
        self.dp_size = self.p_state.dp_size

        # Calculate gradient accumulation steps
        assert args.gradient_accumulation_steps >= 1, \
            f"gradient_accumulation_steps must be >= 1, got {args.gradient_accumulation_steps}"
        assert args.global_batch_size == args.gradient_accumulation_steps * args.micro_batch_size * self.dp_size, \
            f"global_batch_size ({args.global_batch_size}) must be equal to " \
            f"micro_batch_size ({args.micro_batch_size}) * dp_size ({self.dp_size}) * " \
            f"gradient_accumulation_steps ({args.gradient_accumulation_steps})"

    def build_model_and_optimizer(self):
        args = self.args
        self.logger.info("Building model and optimizer...")

        # Build model
        dtype = torch.bfloat16 if args.bf16 and not args.main_params_fp32 else torch.float32
        self.model, self.model_config = build_model(
            args,
            dtype=dtype,
            device=args.init_device,
            initialize_weights=args.init_device != "meta",
        )
        # Model Reproducibility.
        if args.reproduce and hasattr(self.model, "enable_deterministic"):
            self.model.enable_deterministic()
        # Load weights from pretrained checkpoint if specified. We first collect all possible load plans
        # into three groups: before_fsdp_plans, fsdp_plans, after_fsdp_plans.
        # - before_fsdp_plans: (init_device != 'meta') plans that weights loading can be immediately executed.
        #   For example, pretrained submodules, hugging face checkpoints. Loaded by self.model.load_before_fsdp()
        # - fsdp_plans: (init_device == 'meta') plans that weights loading need to be deferred but supported
        #   for fully loading or stream loading. For example, pretrained submodules, hugging face checkpoints.
        #   Loaded by load_and_apply_fsdp2() when applying fsdp.
        # - after_fsdp_plans: plans that weights loading need to be deferred until after fully_shard.
        #   For example, dcp checkpoints, resuming from existing training checkpoints.
        #   Loaded by self.load_after_fsdp()
        self.model.collect_load_plans(
            self.checkpoint_dir, args.load,
            fuse_experts_in_load=args.fuse_experts_in_load,
            copy_mot_in_load=args.copy_mot_in_load and self.model_config.use_mot,
        )
        self.model.load_before_fsdp()

        # Dump the model config
        if self.rank == 0:
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model built with {num_params:,} parameters:")
            for name, module in self.model.named_children():
                num_params = sum(p.numel() for p in module.parameters())
                self.logger.info(f"\t{name:>20s}: {num_params:,}")

            if isinstance(self.model_config, dict):
                # For MoT
                for key, config in self.model_config.items():
                    if config is not None:
                        print_args(f"{key} model config", config)
            else:
                print_args("model config", self.model_config)
            # Dump the model settings to a file.
            dump_configs(self.model_config, self.exp_dir / f"runtime_configs/{self.task_id}_model_config.yaml")
        torch.distributed.barrier()

        # Learning Rate Scheduler
        def lr_scheduler_func(optimizer):
            # TODO: make scheduler type configurable
            lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer=optimizer,
                **args.lr_scheduler_params,
            )
            return lr_scheduler

        if 'muon' in args.optimizer.lower():
            optimizer_factory, optim_param_mapping = build_optimizer_factory_from_model_args(model=self.model, args=args)
        else:
            optimizer_factory = None
            optim_param_mapping = {}

        # clean sd before FSDP
        clean_state_dict = self.model.state_dict()

        # Build Model Engine
        ParallelEngine: Type[HyBaseEngine] = find_engine(args.model_name)     # noqa
        self.logger.info("Start building ParallelEngine...")
        self.model_engine: HyBaseEngine = ParallelEngine(
            model=self.model,
            optimizer_config=dict(
                # `muon` always requires using optimizer_factory to grouping adam params and muon params.
                # Should never fall back to default optimizer_cls. We set it to None to avoid fallback.
                optimizer_cls={'adamw': AdamW, 'adam': AdamW, 'dist_muon': None, 'torch_muon': None, 'muon': None}[args.optimizer.lower()],
                optimizer_kwargs=args.optimizer_params,
                optimizer_factory_for_special_param=optimizer_factory,
                pre_optimizer_hook=pre_optimizer_hook,
            ),
            get_lr_scheduler_func=lr_scheduler_func,
            enable_autocast=args.autocast_dtype not in ["fp32", "float32"],
            autocast_prec=args.autocast_dtype,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            fsdp_gradient_accumulation_steps=getattr(args, "fsdp_gradient_accumulation_steps", None),
            enable_gradient_checkpointing=args.recompute_granularity is not None,
            # Put meta param materialization into init_param_and_apply_fsdp2 for memory-efficient initialization
            # For streaming fsdp implementation, we set initialize_meta_param to false, since meta params will be materialized in apply_fsdp.
            initialize_meta_param=args.fsdp_impl == 'new',
            dp_replicate_param_handler='none',
            activation_offloading=getattr(args, "activation_offloading", False),
            optimizer_offloading=getattr(args, "optimizer_offloading", False),
            enable_compile=args.compile_engine,
        )
        # Try to load from checkpoint if available
        self.load_after_fsdp()
        # Print model structure and parameter summary
        if self.rank == 0:
            # print no weight decay params
            for (optimizer_cls, optimizer_kwargs), param_names in optim_param_mapping.items():
                self.logger.info(f"  {optimizer_cls.__name__} optimizer with kwargs: {optimizer_kwargs} is Applied to:\n{param_names}\n")
                from hy_parallelism.optimizers.muon.torch_muon import Muon
                if optimizer_cls is Muon:
                    for param_name in param_names:
                        if param_name in clean_state_dict and (len(clean_state_dict[param_name].shape) != 2):
                            self.logger.debug(f"{param_name} with shape: {clean_state_dict[param_name].shape} is optimized by Muon.")
            print(self.model_engine.model)
            print(ParameterSummary.from_module(self.model_engine.model).as_table())
            if hasattr(self.model_engine.model, "get_printable_layers"):
                print()
                for layer in self.model_engine.model.get_printable_layers():
                    print(ParameterSummary.from_module(layer).as_table())

        # Build Monitor
        from hymm.utils.monitor import MonitorMaster, TensorBoardConfig, WandbConfig
        tensorboard_config = TensorBoardConfig(
            enabled=args.tensorboard_dir is not None,
            output_path=args.tensorboard_dir,
            job_name=args.task_id,
        )
        wandb_config = WandbConfig(
            enabled=args.wandb_project is not None,
            project=args.wandb_project,
            exp_name=args.wandb_exp_name,
            output_path=args.wandb_dir,
            config=vars(args)
        )
        monitor_config = addict.Dict()
        monitor_config.tensorboard = tensorboard_config
        monitor_config.wandb = wandb_config

        self.model_engine.monitor = MonitorMaster(     # noqa
            monitor_config=monitor_config,
            writer_rank=self.world_size-1,
            wandb_server=args.wandb_server,
        )

        self.logger.info(f"Memory usage after model and optimizer initialization:")
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_mem = torch.cuda.memory_reserved() / (1024 ** 3)
        self.logger.info(f"  Allocated memory: {allocated_mem:.3f} GB")
        self.logger.info(f"  Reserved memory:  {reserved_mem:.3f} GB")
        self.logger.info(f"Training Engine initialized.")

    def build_extra_model(self):
        """ initialize scalar states, denoiser, vae, tkwrapper as global vars. """
        args = self.args

        # Build a fresh scalar_state; it will be overwritten from checkpoint in load_after_fsdp() when resume.
        self.ss = build_scalar_state()

        if args.use_vae:
            self.denoiser = build_denoiser()
            # 离线 latent 训练时 vae_norm_stats_only=True：跳过 VAE 编码器/解码器构建，只加载 normalize 相关参数
            # 用于 normalize 离线 latent（use_vae 保持 True 以维持 latent 空间的模型架构）
            self.vae = build_vae(
                only_encoder=not args.use_vae_decoder,
                norm_stats_only=getattr(args, "vae_norm_stats_only", False),
            )

        self.tkwrapper = build_tkwrapper()

    def build_dataloader(self):
        # Build dataloaders for multimodal tasks
        DatasetsProvider()(None)
        self.combined_iterator = get_combined_iterator()
        self.mm_state = get_mm_state()
        # Apply pack_buffer_state from resume checkpoint if it was deferred in load_after_fsdp()
        pending = None if getattr(self.args, 'no_resume_pack_buffer', False) else getattr(self, '_pending_pack_buffer_state', None)
        if pending is not None:
            state = pending.get("pack_buffer_state")
            loaded_dp_size = pending.get("pack_buffer_dp_size")
            self.combined_iterator.load_pack_buffer_state(
                state, loaded_dp_size=loaded_dp_size
            )
            self._pending_pack_buffer_state = None

    def after_initialize(self):
        args = self.args

        # Define monitored metrics
        loss_names = ["grad_norm", "loss"] + [
            f"{dataset_tag}_text_loss" for dataset_tag in self.mm_state.all_dataset_keys
        ] + [
            f"{dataset_tag}_image_loss" for dataset_tag in self.mm_state.all_dataset_keys
        ] + [
            f"{dataset_tag}_video_loss" for dataset_tag in self.mm_state.all_dataset_keys
        ] + ['global_image_loss', 'global_video_loss', 'global_text_loss']
        if args.moe_aux_loss and args.moe_aux_loss_coeff > 0:
            loss_names.extend(["moe_loss", "capacity_rate"])
            if args.use_mot:
                loss_names.extend(["moe_loss_mot_gen", "capacity_rate_mot_gen"])
        self.loss_names = loss_names

        # Prerun VAE to scan the input sizes for better performance.
        if args.prerun_vae:
            from ..core.extra_model_provider import prerun_vae
            self.model.unshard()
            prerun_vae(self.model)
            self.model.reshard()

    def prepare_model_inputs(self, batch, device):
        with torch.autocast(device_type="cuda", enabled=False):
            model_input_kwargs, bsz, seqlen = prepare_model_inputs(batch, device)
        return model_input_kwargs, bsz, seqlen

    def apply_global_loss_average(self, loss_dict, apply_global_loss_average_fn, batch=None):
        args = self.args

        # Optionally average LM (text) loss over global token count (all DP ranks)
        if getattr(args, "use_global_discrete_loss_average", False):
            apply_global_loss_average_fn(
                sum_key="discrete_loss_sum",
                count_key="discrete_loss_count",
                global_loss_name="global_text_loss",
                loss_dict=loss_dict,
            )
        # Optionally average diffusion loss over global image sample count (all ranks)
        if getattr(args, "use_global_diffusion_loss_average", False):
            apply_global_loss_average_fn(
                sum_key="diff_loss_sum",
                count_key="diff_loss_count",
                loss_weight=loss_dict.pop("diff_loss_weight", args.image_loss_weight),
                global_loss_name="global_image_loss",
                loss_dict=loss_dict,
                gbca_count=None if batch is None else batch.get("micro_batch_image_count"),
            )

    def train_step(self, batch):
        args = self.args
        device = torch.device("cuda", args.local_rank)

        model_input_kwargs, bsz, seqlen = self.prepare_model_inputs(batch, device)

        def loss_closure(model_output, input_args_kwargs_dict):     # noqa
            loss_dict = model_output.losses

            def _apply_global_loss_average(
                sum_key,
                count_key,
                global_loss_name,
                loss_weight=1.0,
                loss_dict={},
                gbca_count=None,
            ):
                all_reduce_dtype = torch.float64

                if sum_key in loss_dict and count_key in loss_dict:
                    loss_sum = loss_dict.pop(sum_key)
                    loss_count = loss_dict.pop(count_key).to(all_reduce_dtype)
                else:
                    # keep all ranks participating in all_reduce to avoid deadlock
                    ref_device = loss_dict["loss"].device
                    loss_sum = torch.tensor(0.0, device=ref_device, dtype=torch.float32)
                    loss_count = torch.tensor(0.0, device=ref_device, dtype=all_reduce_dtype)

                # Log as (local_sum, local_count); scalar_state aggregates Σsum/Σcount,
                # i.e. sample-weighted average across all microbatches and DP ranks.
                loss_dict[global_loss_name] = (loss_sum.detach(), loss_count.detach().clone())

                if gbca_count is not None:
                    # Step-level average count from dp-load-balance; skip per-mb all_reduce.
                    global_count = float(gbca_count)
                else:
                    dist.all_reduce(loss_count, op=dist.ReduceOp.SUM, group=self.p_state.dp_group)
                    global_count = loss_count.item()
                if global_count <= 0 or loss_weight <= 0:
                    return
                global_avg_loss = loss_sum * self.p_state.dp_size / (global_count + 1e-8)
                loss_dict["loss"] = loss_dict["loss"] + loss_weight * global_avg_loss

            self.apply_global_loss_average(
                loss_dict=loss_dict,
                apply_global_loss_average_fn=_apply_global_loss_average,
                batch=batch,
            )

            # Convert _global_metric_losses {name: (sum, count)} into per-task entries that scalar_state.update can accumulate as sum/count pairs.
            global_metrics = loss_dict.pop("_global_metric_losses", {})
            for metric_name, (m_sum, m_count) in global_metrics.items():
                loss_dict[metric_name] = (m_sum.item(), m_count.item())

            loss = loss_dict.pop('loss').float()

            return loss, loss_dict

        self.model_engine.register_loss_closure(loss_closure)

        loss = self.model_engine(**model_input_kwargs)

        if torch.isnan(loss).any():
            self.nan_grad_count += 1
            self.logger.warning(f"NaN loss encountered in rank {self.rank}, total NaN count: {self.nan_grad_count}")

        loss_dict = self.model_engine.get_cached_result("loss_dict")
        loss_dict['loss'] = loss

        consumed_metrics = {
            batch["dataset_tag"][0]: {
                "samples": batch["n_samples"].sum().item(),
                "tokens": bsz * seqlen
            }
        }

        return loss_dict, consumed_metrics

    def training_log(self, loss_reduced, duration, timer: Timer):
        step = self.ss.update_steps
        grad_norm = loss_reduced.pop("grad_norm", -1)   # Use -1 as default if not found.
        log_str = (
            f"iteration {step:>8d}/{self.args.train_iters}"
            f" | grad norm: {grad_norm:>8.3f}"
            f" | speed: {1 / duration:>6.2f} it/s, {duration:>6.2f} s/it"
        )
        summary_events = [
            ("Speed/steps_per_sec", 1 / duration, step),
            ("Speed/seconds_per_step", duration, step),
            ("Gradient/grad_norm", grad_norm, step),
        ]

        for i, lr in enumerate(self.ss.lr, start=1):
            order = '' if i == 1 else f' {i}'
            log_str += f" | learning rate{order}: {lr:.6E}"
            summary_events.append((f"Learning Rate/lr{order}", lr, step))

        for name, loss in loss_reduced.items():
            log_str += f" | {name}: {loss:.6E}"
            summary_events.append((f"Loss/{name}", loss, step))
        for name, samples in self.ss.consumed_samples_total.items():
            log_str += f" | {name} smp: {samples:>14,}({self.ss.consumed_epoch[name]})"
            summary_events.append((f"Consumed Samples/{name}", samples, step))
        for name, tokens in self.ss.consumed_tokens_total.items():
            log_str += f" | {name} tks: {tokens:>18,}"
            summary_events.append((f"Consumed Tokens/{name}", tokens, step))
        for name, state in timer.timers.items():
            log_str += f" | {name}_time: {int(state['elapsed_time'] * 1000):,}ms"

        # Log the training information to the logger.
        self.logger.info(log_str)

        # Log the training information to the monitor.
        self.model_engine.monitor.write_events(summary_events)

    # Resume-state extension hooks. Subclasses need to override
    def _collect_extra_client_state(self) -> dict:
        """Extra fields to merge into the saved ``client_state`` dict."""
        return {}

    def _consume_extra_client_state(self, client_state: dict) -> None:
        """Consume extra fields from ``client_state`` on resume. Called inside ``load_after_fsdp``. """
        return None

    def load_after_fsdp(self):
        for plan in self.model.after_fsdp_plans:
            # Load from pretrained checkpoint
            if plan.source == "dcp":
                default_states = self.model_engine.pre_load_state_dict()
                self.model_engine.load_checkpoint(**plan.metadata)
                self.model_engine.post_load_state_dict(default_states)

            # Resume from existing checkpoint if any.
            elif plan.source == "resume":
                # Load checkpoint
                # -- model, optimizer, lr_scheduler
                load_path, training_states = self.model_engine.load_checkpoint(**plan.metadata)
                load_path_pack_buffer_states = Path(load_path) / "pack_buffer_states"
                # -- scalar states
                if 'client_state' in training_states and 'scalar_state' in training_states['client_state']:
                    # 把 checkpoint 的 scalar_state 加载到 self.ss 中
                    from hymm.trainers.helpers import MultiModalScalarStates
                    new_ss = MultiModalScalarStates.from_pretrained(training_states['client_state']['scalar_state'])
                    for field_name, value in vars(new_ss).items():
                        setattr(self.ss, field_name, value)
                    self.logger.info(f"Loaded scalar states: {self.ss.__class__}:\n{self.ss.serialize(indent=4)}")

                # -- subclass-specific extras (e.g. VAE/audio_VAE torch.Generator states).
                self._consume_extra_client_state(training_states.get('client_state', {}) or {})
                # -- pack_buffer for accurate resume with sequence pack
                # Set --no-resume-pack-buffer to skip loading and start with empty buffer.
                if getattr(self.args, 'no_resume_pack_buffer', False):
                    self.logger.info("Skipping pack_buffer restore (--no-resume-pack-buffer).")
                else:
                    pack_buffer_loaded = False
                    # Per-rank file: load with optional redistribute when dp_size changes
                    metadata = None
                    metadata_file = load_path_pack_buffer_states / "pack_buffer_metadata.pt"
                    if metadata_file.exists():
                        metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
                    saved_dp_size = metadata.get("pack_buffer_dp_size", 1) if metadata else None
                    counts_by_key = metadata.get("counts", {}) if metadata else None

                    if saved_dp_size is not None and saved_dp_size != self.dp_size and counts_by_key:
                        my_state = CombinedBatchIterator.load_pack_buffer_from_per_rank_files_with_redistribute(
                            load_path_pack_buffer_states, metadata, self.rank, self.dp_size
                        )
                        if hasattr(self, 'combined_iterator') and self.combined_iterator is not None:
                            # 单 rank 格式要求 state[key]=[list_of_items]，故包一层
                            self.combined_iterator.load_pack_buffer_state(
                                {k: [v] for k, v in my_state.items()}, loaded_dp_size=1
                            )
                            pack_buffer_loaded = True
                        elif any(my_state.values()):
                            self._pending_pack_buffer_state = {
                                "pack_buffer_state": {k: [v] for k, v in my_state.items()},
                                "pack_buffer_dp_size": 1,
                            }
                            pack_buffer_loaded = True
                        if pack_buffer_loaded:
                            self.logger.info(
                                f"Pack_buffer: restored with redistribute (saved_dp_size={saved_dp_size} -> current dp_size={self.dp_size})."
                            )
                    else:
                        per_rank_file = load_path_pack_buffer_states / f"pack_buffer_rank{self.rank}.pt"
                        if per_rank_file.exists():
                            ckpt = torch.load(per_rank_file, map_location="cpu", weights_only=False)
                            saved_dp_size = ckpt.get("pack_buffer_dp_size", 1)
                            if saved_dp_size != self.dp_size:
                                self.logger.info(
                                    f"Pack_buffer: skipping restore (saved_dp_size={saved_dp_size} != current dp_size={self.dp_size}, no metadata for redistribute)."
                                )
                            else:
                                state = ckpt.get("pack_buffer_state")
                                if state is not None and hasattr(self, 'combined_iterator') and self.combined_iterator is not None:
                                    self.combined_iterator.load_pack_buffer_state(state, loaded_dp_size=1)
                                    pack_buffer_loaded = True
                                elif state is not None:
                                    self._pending_pack_buffer_state = {"pack_buffer_state": state, "pack_buffer_dp_size": 1}
                                    pack_buffer_loaded = True

        # Clear cache to avoid unnecessary memory occupation from loaded states after loading.
        torch.cuda.empty_cache()

    def save_checkpoint(self):
        tag = f"iter_{self.ss.update_steps:07d}"

        # Clear cache to avoid OOM during saving.
        torch.cuda.empty_cache()
        # Sync data iterator state
        if self.args.resume_index_batch_sampler:
            self.combined_iterator.sync_state_dict()

        client_state = dict(
            config=vars(self.args),
            scalar_state=self.ss.serialize()
        )
        # Merge in any subclass-specific extra fields.
        client_state.update(self._collect_extra_client_state())

        if not getattr(self.args, 'save_pack_buffer', False):
            self.logger.info("Skipping pack_buffer save (set --save-pack-buffer to enable).")
        else:
            # Save pack_buffer to per-rank files + metadata file only
            pack_buffer_ckpt = self.combined_iterator.get_pack_buffer_state_for_checkpoint(self.dp_size)
            if pack_buffer_ckpt is not None:
                pack_buffer_dir = self.checkpoint_dir / tag / "pack_buffer_states"
                pack_buffer_dir.mkdir(parents=True, exist_ok=True)
                torch.save(pack_buffer_ckpt, pack_buffer_dir / f"pack_buffer_rank{self.rank}.pt")
                self.logger.info(f"Saved pack_buffer to {pack_buffer_dir / f'pack_buffer_rank{self.rank}.pt'}.")
                metadata = self.combined_iterator.get_pack_buffer_metadata_for_checkpoint(self.dp_size)
                if metadata is not None and self.rank == 0:
                    torch.save(metadata, pack_buffer_dir / "pack_buffer_metadata.pt")
                    self.logger.info(f"Saved pack_buffer to {pack_buffer_dir}.")

        self.model_engine.save_checkpoint(
            save_dir=self.checkpoint_dir,
            tag=tag,
            client_state=client_state,
            save_optimizer_states=not self.args.disable_save_optimizer,
            save_all_ranks_training_states=getattr(
                self.args, "save_all_ranks_training_states", False
            ),
        )

        if self.args.save_hf:
            # TODO(jarvizhang): support distributed saving
            hf_save_dir = self.checkpoint_dir / tag / "hf"
            state_dict = self.model.state_dict()
            self.model.state_dict_to_hf(state_dict, hf_save_dir)

            torch.distributed.barrier()

    def prepare_micro_batches(self, micro_batches, batch):
        micro_batches.append(batch)
        return micro_batches

    def check_dp_load_balance_args(self):
        args = self.args
        dp_load_balance = getattr(args, "dp_load_balance", False)
        use_gbca = getattr(args, "use_global_batch_count_average", False)

        if use_gbca and not dp_load_balance:
            raise ValueError("use-global-batch-count-average requires dp-load-balance.")
        if dp_load_balance and not use_gbca:
            raise ValueError("dp-load-balance requires use-global-batch-count-average.")
        if not dp_load_balance:
            return

        if getattr(args, "broadcast_data", False):
            raise ValueError("dp-load-balance and broadcast-data cannot be enabled together.")

        self.logger.info(
            "Pure-torch dp-load-balance enabled: balancing "
            f"{self.grad_accu_steps} micro-batch(es) per update."
        )
        if self.dp_size <= 1:
            raise ValueError(
                "dp-load-balance requires dp_size > 1, "
                f"got dp_size={self.dp_size}."
            )
        if self.grad_accu_steps == 1:
            raise ValueError(
                "dp-load-balance requires gradient-accumulation-steps > 1, "
                f"got gradient-accumulation-steps={self.grad_accu_steps}."
            )
        flops_proxy = getattr(args, "dp_load_balance_flops_proxy", "legacy_lsq")
        model_name = getattr(args, "model_name", "").lower()
        if "leo" in model_name and flops_proxy == "mfu_aligned":
            raise NotImplementedError(
                "Leo dp-load-balance with flops-proxy=mfu_aligned is not implemented yet."
            )

    def maybe_dp_load_balance_ptm(self, micro_batches):
        args = self.args
        if not getattr(args, "dp_load_balance", False):
            return micro_batches

        model_name = getattr(args, "model_name", "").lower()
        if "leo" in model_name:
            from angelptm.megatron.core.models.leo.dp_load_balance import (
                dp_load_balance_batches,
            )
            flops_proxy = getattr(args, "dp_load_balance_flops_proxy", "legacy_lsq")
            arch = None
            if flops_proxy == "mfu_aligned":
                raise NotImplementedError(
                    "Leo dp-load-balance with flops-proxy=mfu_aligned is not implemented yet."
                )

            return dp_load_balance_batches(
                micro_batches,
                dp_group=self.p_state.dp_group,
                use_global_batch_count_average=getattr(
                    args, "use_global_batch_count_average", False
                ),
                flops_proxy=flops_proxy,
                arch=arch,
            )

        from angelptm.megatron.core.models.gemini.broadcast_utils import (
            dp_load_balance_batches,
        )

        return dp_load_balance_batches(
            micro_batches,
            task_priority_list=args.dp_load_balance_task_priority,
            dp_group=self.p_state.dp_group,
            use_global_batch_count_average=getattr(
                args, "use_global_batch_count_average", False
            ),
        )

    def train(self):
        args = self.args

        self.model_engine.train()
        self.nan_grad_count = 0
        self.ss.current_run_update_steps = 0

        if args.init_save:
            self.save_checkpoint()

        self.check_dp_load_balance_args()

        finished = False
        torch.cuda.synchronize()
        start_time = time.time()
        timer = self.timer

        if getattr(args, "init_validation", False):
            self.model_engine.eval()
            self.validation(timer)
            self.model_engine.train()
            torch.cuda.empty_cache()

        # Training loop
        m_micro_batches = []
        timer.start("prepare_data", sync=True, barrier=True)
        with profiler_context(
                args.profile, self.exp_dir, worker_name=f"Rank_{self.rank}"
        ) as prof:
            for bi, micro_batch in enumerate(self.combined_iterator):
                # Validation before training
                if args.try_first_iter and self.ss.update_steps == 0:
                    self.model_engine.eval()
                    self.validation(timer)
                    self.model_engine.train()
                    torch.cuda.empty_cache()

                micro_batch: dict[str, Any]
                m_micro_batches = self.prepare_micro_batches(m_micro_batches, micro_batch)
                if len(m_micro_batches) < self.grad_accu_steps:
                    continue
                timer.stop("prepare_data")

                balanced_micro_batches = self.maybe_dp_load_balance_ptm(m_micro_batches)
                for batch in balanced_micro_batches:
                    # forward-backward step
                    timer.start("forward", sync=True, barrier=True)
                    loss_dict, consumed_metrics = self.train_step(batch)
                    timer.stop("forward")
                    loss = loss_dict["loss"]
                    timer.start("backward", sync=True, barrier=True)
                    self.model_engine.backward(loss, backward_fn=loss_dict.pop("backward_fn", None))
                    timer.stop("backward")
                    is_update_step = self.model_engine.is_gradient_accumulation_boundary()
                    if is_update_step and args.clip_grad > 0:
                        grad_norm = self.model_engine.clip_grad_norm_(
                            self.model_engine.parameters(), max_norm=args.clip_grad).item()
                    else:
                        grad_norm = -1.0
                    # Accumulate training statistics
                    self.ss.update(loss_dict, consumed_metrics, grad_norm, is_update_step, self.model_engine.get_last_lr())
                    # Update model parameters
                    timer.start("update", sync=True, barrier=True)
                    self.model_engine.step()
                    timer.stop("update")

                    # Update expert bias for MoE routers after optimizer step
                    if is_update_step and getattr(self.model_config, 'moe_enable_router_expert_bias', False):
                        mot_und_frozen = (getattr(self.args, "mot_und_frozen", False) if getattr(self.args, "use_mot", False) else False)
                        mot_gen_frozen = (getattr(self.args, "mot_gen_frozen", False) if getattr(self.args, "use_mot", False) else False)
                        model = self.model_engine.model if hasattr(self.model_engine, 'model') else self.model
                        update_router_expert_bias(
                            model=model if isinstance(model, list) else [model],
                            expert_bias_update_rate=self.model_config.moe_expert_bias_update_rate,
                            enable_expert_bias_zero_mean_update=self.model_config.moe_enable_expert_bias_zero_mean_update,
                            mot_und_frozen=mot_und_frozen,
                            mot_gen_frozen=mot_gen_frozen,
                        )

                # If at the boundary of gradient accumulation
                # Log training information
                if self.ss.update_steps % args.log_interval == 0:
                    timer.start("log", sync=True, barrier=True)
                    # All reduce losses
                    loss_reduced = self.ss.all_reduce(
                        self.loss_names, self.mm_state.all_dataset_keys, self.p_state.dp_group)
                    if self.test_loss_dump_file:
                        self.test_loss_values.append(float(loss_reduced['loss']))

                    # Synchronize cuda to accurately measure training speed
                    torch.cuda.synchronize()
                    duration = time.time() - start_time
                    timer.stop("log")
                    self.training_log(loss_reduced, duration, timer)

                    # Reset running states for next logging interval
                    self.ss.reset_running_states()
                    # Reset monitoring variables
                    start_time = time.time()

                # Clear cached micro batches
                m_micro_batches.clear()

                # Enter stopping routine if max steps reached after this step
                if self.ss.update_steps >= args.train_iters:
                    finished = True

                # Save checkpoint
                if ((self.ss.update_steps % args.save_interval == 0) or finished) and self.test_loss_dump_file is None:
                    self.save_checkpoint()

                # Validation on Training
                if args.val_interval > 0 and self.ss.update_steps % args.val_interval == 0:
                    self.model_engine.eval()
                    self.validation(timer)
                    self.model_engine.train()
                    torch.cuda.empty_cache()

                if prof:
                    prof.step()

                if finished:
                    self.logger.info(f"Finished and breaking loop at step={self.ss.update_steps}.")
                    break

                timer.start("prepare_data", sync=True, barrier=True)

        self.maybe_dump_test_losses()

    @torch.no_grad()
    def validation(self, timer):
        from hymm.samplers.hunyuan_multimodal_sampler import HunyuanMultimodalSampler

        timer.start("validation", sync=True, barrier=True)
        sampler = HunyuanMultimodalSampler(None, self.rank, self.world_size, trainer=self)
        sampler.run_testsets(
            sample_save_base=self.exp_dir / f"samples/iter_{self.ss.update_steps:07d}",
        )
        timer.stop("validation")


    def maybe_dump_test_losses(self) -> None:
        if not self.test_loss_dump_file:
            return
        if self.rank != 0:
            return

        dump_path = Path(self.test_loss_dump_file)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.test_loss_values, dump_path)
        self.logger.info(
            f"Saved test loss list with {len(self.test_loss_values)} steps to {dump_path}."
        )

    def exit(self):
        torch.cuda.empty_cache()
        dist.barrier()
        print(f"[rank {self.rank}] Training is complete. Exiting now.")
        dist.destroy_process_group()

if is_bitwise_align_mode():
    from angelptm.toolkits.multimodal_gen.bitwise_align.torch_patcher import run_patches
    # run_patches() need execute here, after class MultimodalTrainer defined.
    run_patches()
