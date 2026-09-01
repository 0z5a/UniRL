
import sys
import os
import gc
import json
import re
import hashlib
import contextlib
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple, Type
import random
import numpy as np
import time
from loguru import logger as all_rank_logger
from pathlib import Path

from accelerate.utils import set_seed
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm
from PIL import Image
from diffusers.utils import check_min_version

from hymm.core.extra_model_provider import (
    build_denoiser, build_vae,  build_text_encoder, build_scalar_state, build_tkwrapper, build_audio_vae,
)
from hymm.engines import find_engine
from hymm.models import build_model
from hymm.models.diffusion.leo_hf import LeoModelHF
from hymm.models.reward_models.video_rewards import get_reward_fn
from hymm.core.global_vars import get_video_denoiser, get_text_encoder, get_mm_state, get_combined_iterator, get_denoiser
from hymm.models.tokenizers import load_tokenizer
from hymm.trainers.rl.utils.file_utils import validate_video_csv_files, convert_to_json_serializable
from hymm.utils.torch_utils import nanstd, PRECISION_TO_TYPE
from hymm.utils.helpers import default
from hymm.trainers.rl.utils.gc import (aggressive_empty_cache, log_gpu_memory_usage)
from hymm.ar.pipelines.pipeline_leo_rl import Leo2ReFLPipeline, Leo2GRPOPipeline

from hymm.diffusion.schedulers import FlowMatchDiscreteScheduler
from hy_parallelism.utils import sync_object_for_parallel_training, sync_random_states
from hy_parallelism.engines.parallel_engine import BaseParallelEngine
from hymm.trainers.pretrain_pure_torch_dit import Leo2Trainer
from hymm.data_kits.csv_dataset import MessageListDataset
from hymm.data_kits.datasampler import DistributedSamplerFix
from processors.video_kits import save_video_audio
from hymm.utils.helpers import readable_time
from hymm.utils.torch_utils import Timer


from torch.utils.data import DataLoader
from hymm.data_kits.video_prompt_dataset import VideoPromptDataset
from hymm.data_kits.samplers import RepeatRandomDistributedSampler
from hymm.utils.torch_utils import set_worker_seed_builder
from hymm.models.autoencoders.hy.cache_utils import setup_vae_checkpointing
from hymm.models.autoencoders import denormalize_vae_latents
from hy_parallelism.training.checkpointing import eager_offload_context, ACTIVATION_POOL_NAME
from hy_parallelism.training.pinned_memory_pool import get_pinned_memory_pool, has_pinned_memory_pool


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


# --- Leo2.1 sample-input contract ---------------------------------------------
# The new MoE (multi_stream_dit) path makes LeoModelHF.prepare_inputs_for_generation
# dereference these kwargs with hard kwargs["..."] indexing, so every key must be
# present (None is fine, missing is a KeyError). Captured per-sample during rollout
# and re-splatted into the training-time denoise step.
_PER_SAMPLE_TENSOR_KEYS = (
    # attention / RoPE structure
    "attention_mask",
    # for gen image/video
    "visual_mask", "timesteps_index",
    # for gen audio
    "audio_mask",
    # for cond text
    "cond_text_states", "cond_text_mask", "text_mask",
    # sequence packing
    "und_token_indices", "gen_token_indices", "audio_token_indices",
)
_PER_SAMPLE_LIST_KEYS = (
    # attention / RoPE structure
    "rope_media_info",
)

_MMDIT_KWARG_KEYS = _PER_SAMPLE_TENSOR_KEYS + _PER_SAMPLE_LIST_KEYS


def _capture_sample_inputs(model_inputs, idx):

    def take_tensor(key):
        v = model_inputs.get(key, None)
        return v[idx].detach().clone() if isinstance(v, torch.Tensor) else None

    captured = {key: take_tensor(key) for key in _PER_SAMPLE_TENSOR_KEYS}
    # rope_media_info is a list[tuple] per row, not a tensor; slice as a list.
    rope = model_inputs.get("rope_media_info", None)
    captured["rope_media_info"] = [rope[k] for k in idx] if rope is not None else None
    return captured


def _stack_mini_inputs(per_sample_dicts):
    """Stack a list of per-sample input dicts into one mini-batch dict.

    Tensor keys are stacked cfg-major to [cfg*MB, ...]; rope_media_info is
    cfg-major list-flattened to match.
    """
    if not per_sample_dicts:
        return {}
    mb = len(per_sample_dicts)

    def _partition(key):
        vals = [d.get(key) for d in per_sample_dicts]
        non_null = [v for v in vals if v is not None]
        if non_null and len(non_null) != mb:
            raise ValueError(
                f"_stack_mini_inputs: key {key!r} is partially None across the "
                f"mini-batch ({len(non_null)}/{mb} samples filled). All samples in a "
                "mini-batch must agree on whether an mmdit kwarg is present."
            )
        return non_null

    batched = {}
    for key in _PER_SAMPLE_TENSOR_KEYS:
        non_null = _partition(key)
        if not non_null:
            batched[key] = None
            continue
        stacked = torch.stack(non_null, dim=0)                                # [MB, cfg, ...]
        batched[key] = stacked.transpose(0, 1).reshape(-1, *stacked.shape[2:])  # [cfg*MB, ...]
    for key in _PER_SAMPLE_LIST_KEYS:
        non_null = _partition(key)
        if not non_null:
            batched[key] = None
            continue
        cfg = len(non_null[0])
        # cfg-major flatten: [[cond_0, uncond_0], [cond_1, uncond_1], ...]
        #                 -> [cond_0, cond_1, ..., uncond_0, uncond_1, ...]
        batched[key] = [v[c] for c in range(cfg) for v in non_null]
    return batched


def get_post_train_video_dataloader(args, logger, text_encoder, text_encoder_2, dp_degree, dp_rank, local_seed=None):
    if args.post_train_type == "grpo" or args.post_train_type == "refl":
        prompt_column = getattr(args, "prompt_column", None)
        video_dataset = VideoPromptDataset(args, logger, args.train_video_csv, text_encoder, text_encoder_2, prompt_column=prompt_column)
    else:
        raise ValueError(f"Invalid post-train type: {args.post_train_type}")

        # Use RepeatRandomDistributedSampler: split data to each DP rank to execute, reduce peak memory on single card
        # mini_repeat_count should be num_generations so that each prompt is repeated num_generations times
        # and distributed across different ranks for parallel generation
    num_generations = getattr(args, 'num_generations', 1)
    video_batch_size = args.video_micro_batch_size[-1]
    
    # CRITICAL: Validate configuration to ensure RepeatRandomDistributedSampler works correctly with group logic
    # For RepeatRandomDistributedSampler to work correctly with our group logic,
    # video_batch_size and num_generations must satisfy one of these conditions:
    # 1. video_batch_size >= num_generations and video_batch_size % num_generations == 0
    #    -> Each rank has complete group(s), ranks_per_group = 1
    # 2. video_batch_size < num_generations and num_generations % video_batch_size == 0
    #    -> Multiple ranks form a group, ranks_per_group > 1
    #
    # Otherwise, samples within a rank may come from different prompts, breaking group structure!
    if video_batch_size >= num_generations:
        # Case 1: Each rank should have complete group(s)
        if video_batch_size % num_generations != 0:
            raise ValueError(
                f"Invalid configuration: video_batch_size ({video_batch_size}) must be divisible by "
                f"num_generations ({num_generations}) when video_batch_size >= num_generations. "
                f"This ensures each rank processes complete group(s). "
                f"Current remainder: {video_batch_size % num_generations}. "
                f"Please adjust video_micro_batch_size or num_generations."
            )
        ranks_per_group = 1
        num_groups_per_rank = video_batch_size // num_generations
        logger.info(f"[Config Validation] video_batch_size={video_batch_size}, num_generations={num_generations}, "
                   f"ranks_per_group={ranks_per_group}, num_groups_per_rank={num_groups_per_rank}")
    else:
        # Case 2: Multiple ranks should form a group
        if num_generations % video_batch_size != 0:
            raise ValueError(
                f"Invalid configuration: num_generations ({num_generations}) must be divisible by "
                f"video_batch_size ({video_batch_size}) when video_batch_size < num_generations. "
                f"This ensures multiple ranks can form complete group(s). "
                f"Current remainder: {num_generations % video_batch_size}. "
                f"Please adjust video_micro_batch_size or num_generations."
            )
        ranks_per_group = num_generations // video_batch_size
        num_groups_per_rank = 1
        logger.info(f"[Config Validation] video_batch_size={video_batch_size}, num_generations={num_generations}, "
                   f"ranks_per_group={ranks_per_group}, num_groups_per_rank={num_groups_per_rank}")
    
    video_sampler = RepeatRandomDistributedSampler(
        video_dataset,
        num_replicas=dp_degree,
        rank=dp_rank,
        shuffle=True,
        seed=args.global_seed,
        drop_last=True,
        batch_size=video_batch_size,
        repeat_count=1,
        mini_repeat_count=num_generations,  # Each prompt repeated num_generations times, distributed across ranks
    )

    def video_collate_fn(batch):
        """Custom collate that zips tuple fields into lists instead of stacking.
        This prevents default_collate from recursively merging message_list dicts.
        """
        # batch is a list of tuples: [(index, prompt, seed, ref_image_path, message_list, csv_file_name), ...]
        return tuple(list(field) for field in zip(*batch))

    video_dataloader = DataLoader(
        video_dataset,
        batch_size=args.video_micro_batch_size[-1],
        shuffle=False,
        sampler=video_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
        worker_init_fn=set_worker_seed_builder(dp_rank),
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=video_collate_fn,
    )
    logger.info('Dataloader init done')
    return video_dataset, video_sampler, video_dataloader


class Leo2ReFLFSDPTrainer(Leo2Trainer):
    """FSDP (pure torch) ReFL trainer for Leo."""

    def __init__(self, args):
        # leo pipeline requires HF functions
        if not args.model_structure.endswith("HF"):
            args.model_structure += "HF"

        # Debug fast-init: skip checkpoint loading and run a tiny rollout.
        if getattr(args, "_debug_fast_init", False):
            self._debug_fast_init_patch_args(args)
            args.sample_interval = -1
            args.refl_sampling_steps = 3
            args.refl_timestep_t1 = 1
            args.refl_timestep_t2 = 2

        super().__init__(args)

        if not getattr(args, "output_dir", None):
            args.output_dir = str(Path(args.save).parent)
        
        setup_vae_checkpointing(getattr(self, "vae", None), self.args)
        # ReFL backprops reward gradients through the differentiable image
        # resize (CLIP preprocessing) and ViT interpolate_pos_encoding, whose
        # upsample bicubic/bilinear backward kernels have no deterministic CUDA
        # impl. If base_trainer enabled determinism, downgrade it to warn_only
        # so loss.backward() falls back with a warning instead of crashing.
        if getattr(args, "reproduce", False) and torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(True, warn_only=True)

        if args.sample_interval > 0:
            validate_video_csv_files(args, logger=self.logger)

        self.finalize_model()
        

        def prompt_fn(prompt, row=None):
            # FPS:24,
            if args.prompt_prepend_fps:
                prompt = f"FPS:{args.video_fps}:" + prompt
            if args.prompt_prepend_content:
                prompt = args.prompt_prepend_content + prompt
            if args.prompt_append_content:
                prompt = prompt + args.prompt_append_content
            return {"role": "user", "content": prompt}

        self.prompt_fn = prompt_fn
        self.reward_th = args.reward_config["reward_th"]

        # ---- Multi-dimension reward normalization (z-score) + optional EMA ----
        rc = args.reward_config
        self.use_reward_norm = bool(rc.get("use_reward_norm", False))
        self.use_reward_ema = bool(rc.get("use_reward_ema", False))
        self.reward_ema_decay = float(rc.get("reward_ema_decay", 0.99))
        self.reward_ema_warmup_steps = int(rc.get("reward_ema_warmup_steps", 0))
        self.reward_norm_eps = float(rc.get("reward_norm_eps", 1e-6))

        self.reward_ema_mean = {}
        self.reward_ema_sq = {}
        reward_mean_cfg = rc.get("reward_mean", {}) or {}
        reward_std_cfg = rc.get("reward_std", {}) or {}
        if isinstance(reward_mean_cfg, dict):
            for k in reward_mean_cfg:
                m = float(reward_mean_cfg[k])
                s = float(reward_std_cfg.get(k, 1.0))
                self.reward_ema_mean[k] = m
                self.reward_ema_sq[k] = s * s + m * m
        # Number of EMA updates performed so far (drives warmup; persisted on resume).
        self._reward_ema_updates = 0

        # Overwrite generation config with task-specific kwargs if specified
        for key, value in args.t2vi2v_task_kwargs.items():
            if hasattr(self.model.generation_config, key):
                setattr(self.model.generation_config, key, value)

    def _debug_fast_init_patch_args(self, args):
        """Clear checkpoint-related args so `collect_load_plans` produces empty plan
        lists for the main DiT model (and, transitively, the ref model). This skips
        HF/bin/dcp pretrained loads as well as resume-from-iter.
        """
        orig = {
            "load": getattr(args, "load", None),
            "resume": getattr(args, "resume", None),
            "load_pretrained_submodules": getattr(args, "load_pretrained_submodules", None),
        }
        args.load = None
        args.resume = False
        args.load_pretrained_submodules = False
        print(
            "[DEBUG] LEO_DEBUG_FAST_INIT=1: skipping main/ref model checkpoint loads. "
            f"Forced args.load=None (was {orig['load']!r}), "
            f"args.resume=False (was {orig['resume']!r}), "
            f"args.load_pretrained_submodules=False (was {orig['load_pretrained_submodules']!r}). "
            "Main DiT and reference model will run with RANDOM weights."
        )

    def load_after_fsdp(self):
        """Skip main model dcp/resume checkpoint loads in debug fast-init mode."""
        if getattr(self, "_debug_fast_init", False):
            self.logger.warning(
                "[DEBUG] LEO_DEBUG_FAST_INIT=1, skipping main model load_after_fsdp "
                "(dcp pretrained / resume from iter_*)"
            )
            return
        super().load_after_fsdp()

    def init_env(self):
        super().init_env()
        self.dp_degree = self.dp_size
        self.sp_rank = self.p_state.cp_rank
        self.sp_size = self.p_state.cp_size
        self.sp_group = self.p_state.cp_group
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.set_seed()

    def set_seed(self):
        # t within the same seq parallel group should be the same. Noise should be different.
        self.args.local_seed = self.args.global_seed + self.rank // self.p_state.cp_size  # cp 组内 seed 相同
        set_seed(self.args.local_seed)
        random.seed(self.args.local_seed)
        print(f"dp_rank: {self.dp_rank}, rank: {self.rank},  local_seed: {self.args.local_seed}")

    def build_reward_model(self):
        args = self.args
        reward_config = args.reward_config
        params = reward_config.get("params", {})
        if reward_config["reward_model"] == "altclip_rm":
            from hymm.models.reward_models.altclip_rm import AltCLIPRM
            self.reward_inferencer = AltCLIPRM(
                params["model_path"],
                pretrained_model_name_or_path=params["pretrained_model_name_or_path"],
                processor_cache_dir=params["processor_cache_dir"],
                resize_res=params["resize_res"],
            )
        elif reward_config["reward_model"] == "lrm":
            from hymm.models.diffusion import build_model
            self.logger.info("Building LRM reward model...")

            # The LRM reward checkpoint (reward_config["params"]["model_path"]) is a
            # PureTorch DCP already saved in the fsdp2 layout. The ptm2->fsdp2 conversion
            # flags required by the *main* model's SFT checkpoint (key_mapping /
            # load_remove_prefix / swap_gate_and_up) are global args and would otherwise
            # be applied here too, double-swapping / mis-mapping / mis-stripping the LRM
            # weights. Temporarily neutralize them (mirrors the model_structure swap) and
            # load from the dedicated reward checkpoint instead of args.load.
            reward_ckpt = params["model_path"]
            saved_args = dict(
                model_structure=args.model_structure,
                key_mapping=args.key_mapping,
                load_remove_prefix=args.load_remove_prefix,
                swap_gate_and_up=args.swap_gate_and_up,
            )
            try:
                args.model_structure = "LeoModelLRM"  # for replace
                args.key_mapping = None
                args.load_remove_prefix = False
                args.swap_gate_and_up = False

                dtype = torch.bfloat16 if args.bf16 and not args.main_params_fp32 else torch.float32
                self.reward_inferencer, _ = build_model(
                    args,
                    dtype=dtype,
                    device=args.init_device,
                    initialize_weights=False,
                )

                self.reward_inferencer.collect_load_plans(
                    self.checkpoint_dir, reward_ckpt,
                    fuse_experts_in_load=args.fuse_experts_in_load,
                    copy_mot_in_load=args.copy_mot_in_load and self.model_config.use_mot,
                )
                self.reward_inferencer.load_before_fsdp()

                ParallelEngine: Type[BaseParallelEngine] = find_engine(args.model_name)  # noqa
                self.reward_inferencer_engine: BaseParallelEngine = ParallelEngine(
                    model=self.reward_inferencer,
                    enable_autocast=args.autocast_dtype not in ["fp32", "float32"],
                    autocast_prec=args.autocast_dtype,
                    initialize_meta_param=False,
                    dp_replicate_param_handler='none',
                )

                for plan in self.reward_inferencer.after_fsdp_plans:
                    if plan.source == "dcp":
                        default_states = self.reward_inferencer_engine.pre_load_state_dict()
                        self.reward_inferencer_engine.load_checkpoint(**plan.metadata)
                        self.reward_inferencer_engine.post_load_state_dict(default_states)

                self.reward_inferencer.eval()
                for param in self.reward_inferencer.parameters():
                    param.requires_grad = False
            finally:
                args.model_structure = saved_args["model_structure"]
                args.key_mapping = saved_args["key_mapping"]
                args.load_remove_prefix = saved_args["load_remove_prefix"]
                args.swap_gate_and_up = saved_args["swap_gate_and_up"]

            self.logger.info(f"--> LRM reward model loaded from {reward_ckpt}")

        elif reward_config["reward_model"] == "qwen_rm":
            from hymm.models.reward_models.hyvideo_local_reward import HyVideoRewardLocal
            self.reward_inferencer = HyVideoRewardLocal(model_config=reward_config["model_config"], model_path=reward_config.get("model_path", None))


            # Freeze RM params: RM is inference-only, no optimizer updates.
            # Gradients still flow through activations back to DiT (chain rule on inputs).
            for p in self.reward_inferencer.model.parameters():
                p.requires_grad = False

            # Enable gradient checkpointing for all layers (vision + text decoder).
            # Requires train() mode for HF's GradientCheckpointingLayer to activate.
            # Safe: Qwen3-VL has attention_dropout=0.0 and no BatchNorm.
            self.reward_inferencer.model.train()
            self.reward_inferencer.model.config.use_cache = False
            self.reward_inferencer.model.gradient_checkpointing_enable()

            # The Qwen reward model is a vanilla HF model (eager from_pretrained,
            # safetensors) -- it has none of the internal DCP / ParallelEngine
            # hooks the LRM/ref paths use. Weights are already materialized on
            # every rank, so we only need to FSDP2-shard the loaded module
            # (mirrors how the frozen HF text encoder is sharded).
            if reward_config.get("apply_fsdp", True):
                self.reward_inferencer.apply_fsdp()

            self.logger.info(f"--> Qwen reward model loaded from {reward_config['model_config']}")
        else:
            raise ValueError(f"Invalid reward model: {reward_config['reward_model']}")

        log_gpu_memory_usage(
            f"[memory] after loading reward model {reward_config['reward_model']}", logger=self.logger
        )


    def build_extra_model(self):
        
        self.logger.info('--> load extra models')
        args = self.args

        # ss maybe loaded from checkpoint, so only build when not exist.
        if not hasattr(self, "ss") or self.ss is None:
            self.ss = build_scalar_state()

        # vae, text
        self.vae = build_vae(only_encoder=False)
        self.audio_vae = build_audio_vae() if args.use_audio_vae else None

        # denoiser
        self.denoiser = build_denoiser()
        
        self.text_encoder = build_text_encoder()
        self.text_encoder_2 = None
        log_gpu_memory_usage("[memory] after loading vae and text encoder model", logger=self.logger)

        # reward model
        self.build_reward_model()
        # The Qwen RM (HyVideoRewardLocal) manages its own device placement
        # (device_map at load) and FSDP2 sharding internally, and is already
        # eval()'d. The generic .to(self.device) below would be redundant and
        # can interfere with the FSDP2-sharded DTensor params, so skip it. Besides,
        # the Qwen RM (HyVideoRewardLocal) needs train mode for gradient_checkpointing_enable().
        if self.args.reward_config["reward_model"] != "qwen_rm":
            self.reward_inferencer.eval()
            self.reward_inferencer.to(self.device)

        log_gpu_memory_usage("[memory] after loading reward model", logger=self.logger)


        # load ref model for kl
        if args.kl_weight > 0:
            self.load_ref_model()
        else:
            self.ref_model = None
            self.logger.info("KL regularization is disabled (kl_weight=0 or ref_model=None)")

        self.tokenizer = load_tokenizer(args.tokenizer_name, args.tokenizer_class)
        
    def after_initialize(self):
        pass

    def _collect_extra_client_state(self) -> dict:
        state = super()._collect_extra_client_state()
        # Number of data batches consumed in the current epoch, for exact
        # data-position resume (used together with the base-restored self.ss.epoch).
        state["refl_epoch_consumed_batches"] = getattr(self, "_epoch_consumed_batches", 0)
        # Persist reward-normalization EMA stats so resume continues seamlessly.
        state["reward_ema_mean"] = dict(getattr(self, "reward_ema_mean", {}))
        state["reward_ema_sq"] = dict(getattr(self, "reward_ema_sq", {}))
        state["reward_ema_updates"] = int(getattr(self, "_reward_ema_updates", 0))
        return state

    def _consume_extra_client_state(self, client_state: dict) -> None:
        super()._consume_extra_client_state(client_state)
        # Stash consumed-batches count; applied at the top of the resumed epoch in
        # train() together with self.ss.epoch (restored by the base class).
        if client_state and "refl_epoch_consumed_batches" in client_state:
            self._resumed_epoch_consumed_batches = int(client_state.get("refl_epoch_consumed_batches", 0))
            self.logger.info(
                f"Staged ReFL epoch_consumed_batches for resume: {self._resumed_epoch_consumed_batches}"
            )
        # Restore reward-normalization EMA stats (only when EMA is enabled; static
        # config values are used otherwise). Missing keys => keep config init.
        if getattr(self, "use_reward_ema", False) and client_state:
            saved_mean = client_state.get("reward_ema_mean")
            saved_sq = client_state.get("reward_ema_sq")
            if isinstance(saved_mean, dict) and isinstance(saved_sq, dict) and saved_mean:
                self.reward_ema_mean = dict(saved_mean)
                self.reward_ema_sq = dict(saved_sq)
                self._reward_ema_updates = int(client_state.get("reward_ema_updates", 0))
                self.logger.info(
                    f"Restored reward EMA stats (updates={self._reward_ema_updates}): "
                    f"mean={self.reward_ema_mean}, sq={self.reward_ema_sq}"
                )

    def build_dataloader(self):
        self.video_dataset, self.video_sampler, self.video_loader = get_post_train_video_dataloader(self.args, self.logger, self.text_encoder, self.text_encoder_2, self.dp_size, self.dp_rank, local_seed=self.args.local_seed)

    def finalize_model(self):
        self.model.train()
        self.model.load_generation_config(default(self.args.generation_config, self.args.ckpt))
        self.logger.info(f"Generation config: {self.model.generation_config}")
        # since txt_hidden_states output is not used in loss computation, gradients would be None for these params anyway.
        # Filter them out to keep dist_muon optimizer happy (zeros_like requires Tensor, not None).
        last_layer_idx = self.model_config['main_branch'].num_layers - 1
        filter_out_params = {
            f"layers.{last_layer_idx}.self_attn.o_proj_txt.weight",
            f"layers.{last_layer_idx}.self_attn.o_proj_txt.bias",
            f"layers.{last_layer_idx}.mlp_txt.gate_and_up_proj.weight",
            f"layers.{last_layer_idx}.mlp_txt.down_proj.weight",
        }
        for name, p in self.model.named_parameters():
            if name in filter_out_params:
                p.requires_grad = False

        # Initialize vae, tokenizer
        self.model.tokenizer = build_tkwrapper()
        if self.args.use_vae:
            self.model.model_dict['vae'] = self.vae
            self.model.model_dict['text_encoder'] = self.text_encoder
        if self.args.use_audio_vae:
            self.model.model_dict['audio_vae'] = self.audio_vae

    def load_ref_model(self):
        args = self.args
        if not args.kl_weight > 0:
            self.ref_model = None
            self.logger.info("KL regularization is disabled (kl_weight=0 or ref_model=None)")
            return None

        dtype = torch.bfloat16 if args.bf16 and not args.main_params_fp32 else torch.float32
        self.model_dtype = dtype
        self.ref_model, _ = build_model(
            args,
            dtype=dtype,
            device=args.init_device,
            initialize_weights=False,
        )
        assert isinstance(self.ref_model, LeoModelHF), "Model building function must return a hymm.models.diffusion.LeoModelHF instance."
        if getattr(self, "_debug_fast_init", False):
            self.logger.warning(
                "[DEBUG] LEO_DEBUG_FAST_INIT=1, building reference model with RANDOM weights "
                "(checkpoint loading is skipped via patched args.load/args.resume)"
            )
        else:
            self.logger.info("--> loading reference model")

        # The KL reference model must ALWAYS load the original (frozen) pretrained
        orig_resume = args.resume
        args.resume = False
        try:
            self.ref_model.collect_load_plans(
                self.checkpoint_dir, args.load,
                fuse_experts_in_load=args.fuse_experts_in_load,
                copy_mot_in_load=args.copy_mot_in_load and self.model_config.use_mot,
            )
        finally:
            args.resume = orig_resume
        self.ref_model.load_before_fsdp()

        ParallelEngine: Type[BaseParallelEngine] = find_engine(args.model_name)  # noqa
        self.ref_model_engine: BaseParallelEngine = ParallelEngine(
            model=self.ref_model,
            enable_autocast=args.autocast_dtype not in ["fp32", "float32"],
            autocast_prec=args.autocast_dtype,
            initialize_meta_param=args.fsdp_impl == 'new',
            dp_replicate_param_handler='none',
        )

        for plan in self.ref_model.after_fsdp_plans:
            if plan.source == "dcp":
                default_states = self.ref_model_engine.pre_load_state_dict()
                self.ref_model_engine.load_checkpoint(**plan.metadata)
                self.ref_model_engine.post_load_state_dict(default_states)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.logger.info("--> reference model loaded")
        log_gpu_memory_usage("[memory] after loading reference model", logger=self.logger)

        kl_compute_mode = getattr(args, "kl_compute_mode", "rollout_phase")
        self.logger.info(f"KL computation mode: {kl_compute_mode}")
        if kl_compute_mode == "rollout_phase":
            self.logger.info("  -> Pre-computing Reference Model statistics during rollout (saves memory during training)")
        else:
            self.logger.info("  -> Computing Reference Model statistics on-the-fly during training")
    

    def build_validation_dataset(self):
        from hymm.data_kits.video_prompt_dataset import VideoPromptDataset
        return VideoPromptDataset(self.args, self.logger, self.args.video_csv)

    @torch.no_grad()
    def sample_validation(self):
        args = self.args 
        run_task_kwargs = args.t2vi2v_task_kwargs
        index_task_kwargs = args.t2vi2v_index_kwargs

        # switch to evaluation mode
        self.model_engine.eval()

        # Restore the text encoder to GPU before generating.
        if getattr(args, "text_encoder_offload", False):
            device = self.device
            for te in (self.text_encoder, self.text_encoder_2):
                if te is not None and hasattr(te, 'model'):
                    te.model = te.model.to(device)
                    te.device = device
            torch.cuda.empty_cache()
            self.logger.info(f"Rank {self.rank}: Text encoders loaded to GPU for sample_validation")

        # Overwrite generation config with task specific kwargs if specified.
        for key, value in run_task_kwargs.items():
            if hasattr(self.model.generation_config, key):
                setattr(self.model.generation_config, key, value)

        self.model.build_diffusion_pipeline()

        video_dirs = []
        save_dir = os.path.join(args.output_dir, "samples", f"{self.ss.update_steps:07d}")
        os.makedirs(save_dir, exist_ok=True)
        for testset in index_task_kwargs['testsets']:
            dataset = MessageListDataset(
                testset,
                save_dir,
                tokenizer=self.tokenizer,
                prompt_fn=self.prompt_fn
            )
            sampler = DistributedSamplerFix(dataset, num_replicas=self.p_state.dp_size,
                                            rank=self.p_state.dp_rank, shuffle=False, drop_last=False,
                                            add_extra_samples="extend")
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler,
                                    drop_last=False, collate_fn=getattr(dataset, "collate_fn", None))
            save_base = dataset.save_dir

            timer = Timer(enabled=True)

            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 1: break
                self.logger.info(f"Generating batch {batch_idx + 1} / {len(dataloader)} ...")
                timer.start(f"Batch")
                message_list=batch[dataset.name_mapper("message_list")]

                num_frames = run_task_kwargs["num_frames"]
                nf_col = dataset.name_mapper("num_frames")
                if nf_col in batch and batch[nf_col] is not None and len(batch[nf_col]) > 0:
                    nf_val = batch[nf_col][0]
                    if nf_val is not None and str(nf_val).strip().lower() not in ("", "nan", "none"):
                        try:
                            num_frames = int(float(nf_val))
                        except (ValueError, TypeError):
                            self.logger.warning(f"[sample_validation] invalid num_frames={nf_val!r}, using task default {num_frames}")
                self.logger.info(f"[sample_validation] num_frames={num_frames} (task default={run_task_kwargs['num_frames']})")
                outputs = self.model.generate_video(
                    message_list=message_list, seed=batch["seed"], 
                    video_size=run_task_kwargs["image_size"], 
                    num_frames=num_frames, 
                    video_fps=run_task_kwargs["video_fps"], 
                    ref_mode=run_task_kwargs["ref_mode"], 
                    output_type=dict(visual="np", audio="np" if self.model.generation_config.bot_task == "av" else None), 
                    bot_task=run_task_kwargs["bot_task"], 
                    verbose=1 if self.rank == 0 else 0
                )
                outputs = outputs.postprocess_outputs(batch)
                if self.p_state.cp_rank == 0:
                    outputs.save_to(
                        save_base=save_base,
                        summary_file_name=f"results/results_{self.p_state.dp_rank}.csv",
                        fps=run_task_kwargs['video_fps'],
                        sample_rate=args.audio_sample_rate,
                    )
                video_dir = os.path.join(save_base, "videos")
                # Log time
                timer.stop(f"Batch")
                self.logger.info(f"[Task {testset}] "
                        f"[{batch_idx + 1} / {len(dataloader)}] "
                        f"| {readable_time(timer, 'Batch', len(dataloader) - batch_idx - 1)} "
                        f"save to {save_dir}")
            video_dirs.append(video_dir)

        self.model_engine.train()
        return video_dirs

    @torch.no_grad()
    def prepare_samples_online(self, model, batch, global_step, dp_rank, sp_rank, sp_group, sample_step):
        """
        Prepare samples for online ReFL training (Rollout Phase).
        
        This function performs the complete rollout process:
        1. Generate samples using the current policy model
        2. Store generated samples to disk
        Args:
            model: Current policy model (for sample generation)
            batch: Input batch containing prompts, seeds, etc.
            global_step: Current training step
            dp_rank: Data parallel rank
            sp_rank: Sequence parallel rank (within SP group)
            sp_group: Sequence parallel groups
            sample_step: Sample step for generation
        Returns:
            all_latents: All latent states [Batch, Steps+1, C, H, W]
            all_latents_audio: All audio latent states [Batch, Steps+1, ...] for av runs, else None
            all_sample_inputs: List of per-sample MoE sample-input contract dicts
            all_channel_cond: List of per-sample channel condition dicts
            all_save_paths: List of save paths for each sample
            sigma_schedule: Noise schedule used for generation
        """
        args = self.args
        device = self.device

        # ========================================================================
        # Phase 0.0: Move Text Encoder back to GPU (if previously offloaded)
        # ========================================================================
        text_encoder_offload = getattr(args, "text_encoder_offload", False)
        if text_encoder_offload:
            if self.text_encoder is not None and hasattr(self.text_encoder, 'model'):
                self.text_encoder.model = self.text_encoder.model.to(device)
                self.text_encoder.device = device
          
            torch.cuda.empty_cache()
            self.logger.info(f"Rank {self.rank}: Text encoders loaded to GPU")
        
        # ========================================================================
        # Phase 1: Parse Input Batch and Initialize Parallel Group Configuration
        # ========================================================================
        indexs, prompts, seeds, ref_image_paths, message_lists, *_rest = batch
        all_rank_logger.info(f"dp_rank {dp_rank}, sp_rank {sp_rank} prepare samples for indexs {indexs}, prompts len {len(prompts)}")
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Normalize prompts/indexs to list format
        if isinstance(prompts, str):
            prompts = [prompts]
            indexs = [indexs] if not isinstance(indexs, list) else indexs
            message_lists = [message_lists]
        
        video_batch_size = len(prompts)
        
        # Get reference image path (use first one if available, all samples in batch share the same ref image)
        ref_image_path = None
        if isinstance(ref_image_paths, list) and len(ref_image_paths) > 0:
            ref_image_path = ref_image_paths[0]
        
        # ------------------------------------------------------------------------
        # Configure Parallel Group Mode
        # ------------------------------------------------------------------------
        # Key concept: ranks_per_group determines how many ranks collaborate on one group
        # Configuration validation is done in get_post_train_video_dataloader() to catch errors early
        generation_mode = "parallel_groups"
        samples_per_rank = video_batch_size
        
        # Calculate group division (validation already done in dataloader initialization)
        if video_batch_size >= args.num_generations:
            # Case 1: Each rank has complete group(s) - no cross-rank communication needed
            ranks_per_group = 1
            num_groups_per_rank = video_batch_size // args.num_generations
        else:
            # Case 2: Multiple ranks form a group - cross-rank gather needed for rewards
            ranks_per_group = args.num_generations // video_batch_size
            num_groups_per_rank = 1
        
        # Calculate group indices for this rank
        group_idx = dp_rank // ranks_per_group  # Which group this rank belongs to
        rank_in_group = dp_rank % ranks_per_group  # Position within the group
        
        self.logger.info(f"[ParallelGroups] Rank {rank}, dp_rank {dp_rank}: "
                f"video_batch_size={video_batch_size}, num_generations={args.num_generations}, "
                f"ranks_per_group={ranks_per_group}, num_groups_per_rank={num_groups_per_rank}, "
                f"group_idx={group_idx}, rank_in_group={rank_in_group}")
        
        # ------------------------------------------------------------------------
        # Phase 2: Process Seeds for Reproducible Generation
        # ------------------------------------------------------------------------
        # Each sample needs a unique seed to ensure diversity within groups
        processed_seeds = []
        for sidx in range(video_batch_size):
            base_seed = seeds[sidx].item() if isinstance(seeds[sidx], torch.Tensor) else seeds[sidx]
            if args.use_same_noise:
                # Use same seed for all samples (for debugging/testing)
                processed_seeds.append(base_seed)
            else:
                # Each sample gets unique seed based on its position within the group
                if ranks_per_group == 1:
                    # Single-rank group: use local sample index within group
                    sample_idx_in_group = sidx % args.num_generations
                else:
                    # Multi-rank group: offset by rank position to ensure uniqueness across ranks
                    sample_idx_in_group = rank_in_group * video_batch_size + sidx
                processed_seeds.append(base_seed + sample_idx_in_group)
        seeds = processed_seeds
        
        # ========================================================================
        # Phase 3: Initialize Pipeline and Scheduler
        # ========================================================================
        infer_flow_shift = args.infer_flow_shift_video 
        scheduler = FlowMatchDiscreteScheduler(
                    shift=infer_flow_shift,
                    reverse=True,
                    solver="euler",
                )
        scheduler.set_timesteps(num_inference_steps=args.refl_sampling_steps, device=device)
        pipeline = self.create_pipeline(model, scheduler)
        sigma_schedule = scheduler.sigmas  # Store schedule for later use in training

        # ========================================================================
        # Phase 4: Initialize Storage Containers
        # ========================================================================
        all_latents = []      # Store all latent states: [Batch, Steps+1, C, H, W]
        all_latents_audio = []  # Optional audio latent states: [Batch, Steps+1, ...] (empty when audio disabled)
        all_sample_inputs = []  # Store the per-sample MoE sample-input contract (for training)
        all_channel_cond = []   # Store per-sample channel condition images (for training)
        all_save_paths = []    # Store save paths for each sample

        # Ensure main model is on GPU before starting the loop
        if self.model_engine is not None:
            # call model_engine.cuda() protects param attr such as _muon_split_fn, _muon_merge_fn from being deleted
            self.model_engine.cuda()
        else:
            model = model.to(device)
        
        # Get mini-batch size for rollout generation (to manage memory)
        mini_batch_size = getattr(args, 'mini_batch_size_per_rollout', 1)
        if mini_batch_size <= 0:
            mini_batch_size = 1
        
        # ========================================================================
        # Phase 5: Rollout Loop - Generate Samples with Current Policy
        # ========================================================================
        # Process prompts in mini-batches to manage memory usage
        num_prompts = len(prompts)
        for batch_start_idx in range(0, num_prompts, mini_batch_size):
            # --------------------------------------------------------------------
            # 5.1: Prepare Mini-Batch
            # --------------------------------------------------------------------
            batch_end_idx = min(batch_start_idx + mini_batch_size, num_prompts)
            batch_prompts = prompts[batch_start_idx:batch_end_idx]
            batch_message_lists = message_lists[batch_start_idx:batch_end_idx]
            batch_indices = list(range(batch_start_idx, batch_end_idx))
            batch_size_actual = len(batch_prompts)
            
            # Prepare batch seeds, generators, and save paths
            batch_seeds = []
            batch_generators = []

            # For each sample in the mini-batch, prepare generation metadata
            for local_idx, sidx in enumerate(batch_indices):
                seed_value = seeds[sidx] if isinstance(seeds[sidx], (int, float)) else seeds[sidx].item()
                batch_seeds.append(seed_value)
                
                save_dir = os.path.join(args.output_dir, "rl_samples", f"{global_step:07d}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"index_{indexs[sidx]}_seed_{seed_value}_rank_{dp_rank}_sample.mp4")
                all_save_paths.append(save_path)
                batch_generators.append(torch.Generator(device=device).manual_seed(seed_value))
            

            # --------------------------------------------------------------------
            # 5.2: Generate Videos with Current Policy Model
            # --------------------------------------------------------------------
            # Use the task-configured frame count (same source that overrode generation_config in
            # __init__ and that prepare_model_inputs uses), so the rollout latents and the visual_mask
            # agree on the temporal length. Mirrors GRPO's target_length.
            target_length = args.t2vi2v_task_kwargs.get("num_frames", args.num_frames)
            target_size = {256: (192, 336), 480: (352, 624), 640: (480, 848), 720: (544, 960), 960: (720, 1280), 1440: (1080, 1920)}
            target_height, target_width = target_size[args.video_bucket_hw_base_size]

            self.logger.info(f"Rank {rank} generating batch {batch_start_idx//mini_batch_size + 1}/{(num_prompts + mini_batch_size - 1)//mini_batch_size} "
                    f"(samples {batch_start_idx+1}-{batch_end_idx}/{num_prompts}), "
                    f"eta: {args.eta}, batch_size: {batch_size_actual}, "
                    f"{target_length}x{target_height}x{target_width}, flow_shift: {infer_flow_shift}, infer_steps: {args.refl_sampling_steps}")
            
            # Configure determistic sampling for progressive training
            # For progressive training: use SDE (deterministic=False) for trainable timesteps, ODE (deterministic=True) for others
            # For "all" strategy (no mixgrpo): use SDE (deterministic=False) for all timesteps
            determistic = None
            
            with torch.no_grad():
                # Batch generation: pass list of prompts to pipeline for parallel processing
                # Handle generator: use list if batch_size > 1, single generator if batch_size == 1
                generator_arg = batch_generators if batch_size_actual > 1 else batch_generators[0]
                batch_latents_batch, batch_latents_audio_batch, batch_sample_inputs, batch_condition_embeds_list = self.rollout_pipeline(
                    pipeline,
                    model,
                    batch_prompts=batch_prompts,
                    batch_message_lists=batch_message_lists,
                    batch_seeds=batch_seeds,
                    target_height=target_height,
                    target_width=target_width,
                    target_length=target_length,
                    generator_arg=generator_arg,
                    determistic=determistic,
                    sample_step=sample_step
                )
            
            # --------------------------------------------------------------------
            # 5.5: Store Generated Results
            # --------------------------------------------------------------------
            # Store results for each sample in the mini-batch
            for local_idx in range(batch_size_actual):
                all_latents.append(batch_latents_batch[local_idx:local_idx+1])      # [1, Steps+1, C, H, W]
                all_sample_inputs.append(batch_sample_inputs[local_idx])            # MoE sample-input contract dict
                all_channel_cond.append(batch_condition_embeds_list[local_idx])     # channel cond dict
                # Audio latents produced only on av runs; needed for the JOINT forward during the
                # training-time denoise recompute (driven by use_audio_vae).
                if batch_latents_audio_batch is not None:
                    all_latents_audio.append(batch_latents_audio_batch[local_idx:local_idx+1])  # [1, Steps+1, ...]

        
        # Concatenate all collected results into tensors
        all_latents = torch.cat(all_latents, dim=0)      # [Total_Batch, Steps+1, C, T, H, W]
        all_latents_audio = torch.cat(all_latents_audio, dim=0) if len(all_latents_audio) > 0 else None

        # Clean up pipeline to free memory before reference model computation
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # ========================================================================
        # Phase 6.0: Offload Text Encoder to CPU (if enabled) TODO: 这里需要优化一下
        # ========================================================================
        text_encoder_offload = getattr(args, "text_encoder_offload", False)
        if text_encoder_offload:
            if self.text_encoder is not None and hasattr(self.text_encoder, 'model'):
                self.text_encoder.model = self.text_encoder.model.to('cpu')
                self.text_encoder.device = 'cpu'
            if self.text_encoder_2 is not None and hasattr(self.text_encoder_2, 'model'):
                self.text_encoder_2.model = self.text_encoder_2.model.to('cpu')
                self.text_encoder_2.device = 'cpu'
            torch.cuda.empty_cache()
            self.logger.info(f"Rank {rank}: Text encoders offloaded to CPU")
        
        return all_latents, all_latents_audio, all_sample_inputs, all_channel_cond, all_save_paths, sigma_schedule
    

    def refl_denoise_step(
        self,
        message_lists,
        transformer,
        latents,
        timesteps,
        sigma_schedule,
        *,
        channel_cond_vae_images=None,
        audio_latents=None,
        audio_timesteps=None,
        **model_input_kwargs,
    ):
        """Run a single ReFL denoise step of the policy / reference model.

        ``model_input_kwargs`` carries the per-sample MoE sample-input contract
        (see ``_capture_sample_inputs`` for the field list -- they match
        ``LeoModel.forward`` 1-to-1), already stacked cfg-major to ``[cfg*MB, ...]``
        by ``_stack_mini_inputs``. Tensor entries are moved to device and ``None``
        entries are forwarded as-is (NOT dropped) so that
        ``LeoModelHF.prepare_inputs_for_generation``'s ``kwargs[...]`` dereferences
        still hit on the MoE (multi_stream_dit) path. ``channel_cond_vae_images``
        is the GRPO-style channel condition captured during rollout (None for t2v).
        """
        args = self.args
        transformer.train()  # ref_model 和 policy_model 都需要 train mode
        device = transformer.device
        scheduler = build_refl_scheduler(args, sigma_schedule, device)

        # Use rollout_cfg_scale to align rollout / recompute guidance.
        guidance_scale = getattr(args, "rollout_cfg_scale", 6.0)
        pipeline = self.create_pipeline(transformer, scheduler)

        scheduler_timesteps = timesteps.to(device=device, dtype=torch.float32)

        # Move tensor mmdit kwargs to device; forward None entries as-is so
        # prepare_inputs_for_generation's kwargs[...] dereferences still hit.
        model_kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in model_input_kwargs.items()
        }
        model_kwargs["channel_cond_vae_images"] = (
            channel_cond_vae_images.to(device) if channel_cond_vae_images is not None else None
        )

        # Joint av forward: feed the rollout audio latents at the trained step so the joint transformer
        # sees the same audio conditioning it did during rollout. Audio has its own scheduler / per-step
        # timesteps; pipeline.denoise_step uses self.audio_scheduler.scale_model_input internally.
        if audio_latents is not None:
            audio_latents = audio_latents.to(device)
            audio_scheduler_timesteps = audio_timesteps.to(device=device, dtype=torch.float32)
        else:
            audio_scheduler_timesteps = None

        # Step is not used currently, see ClassifierFreeGuidance. Only the visual prediction (main_pred)
        # is returned/used for the ReFL reward; the audio branch runs as a frozen input.
        model_pred, _ = pipeline.denoise_step(
            latents, scheduler_timesteps, 0, model_kwargs, guidance_scale=guidance_scale,
            audio_latents=audio_latents, audio_timestep=audio_scheduler_timesteps,
        )

        return model_pred, scheduler, scheduler_timesteps, pipeline, model_kwargs

    
    def train_one_step(self, model, ref_model, reward_inferencer, sp_rank, sp_group, sp_size, dp_size,
                                batch, device, dp_rank):
        """
        Execute one training step of GRPO (Group Relative Policy Optimization).
        
        This function performs:
        1. Online sample generation (rollout)
        2. Advantage computation with reward normalization
        3. Sample shuffling and timestep permutation
        4. Nested training loops (samples x timesteps)
        5. Policy loss computation with optional KL regularization
        """
        # sp group is equal tp group for megatron
        args = self.args
        world_size = self.world_size
        logger = self.logger
        rank = dist.get_rank() if dist.is_initialized() else 0
        indexs, prompts, seeds, ref_image_paths, message_lists, *_rest = batch


        # ==================== Phase 0: Per-Rank Random Timestep Sampling ====================
        # get refl scheduler
        if self.args.reward_config["use_lrm"]:
            infer_flow_shift = args.infer_flow_shift_video 
            scheduler = FlowMatchDiscreteScheduler(
                        shift=infer_flow_shift,
                        reverse=True,
                        solver="euler",
                    )
            scheduler.set_timesteps(num_inference_steps=args.refl_sampling_steps, device=device)
            timesteps_candidates = scheduler.timesteps.detach().cpu().tolist()

            denoiser = get_denoiser()
            empty_latents = torch.zeros((1)).to(device)
            t_tensor = sync_random_tensor(
                generator_fn=lambda: denoiser.sample(empty_latents)[0],
                shape=(1,),
                dtype=torch.float32,
                device=device,
                sp_group=sp_group,
            )
            t_raw = t_tensor.item() * scheduler.num_train_timesteps
            # Snap to the nearest discrete timestep and return its index in the schedule
            sample_step = min(range(len(timesteps_candidates)), key=lambda i: abs(timesteps_candidates[i] - t_raw))

            if sample_step >= args.refl_sampling_steps:
                sample_step = args.refl_sampling_steps - 1
            elif sample_step <= 1:
                sample_step = 1
            orig_sample_step = sample_step

            sample_step_tensor = torch.tensor([sample_step], dtype=torch.long, device=device)
            gathered = [torch.zeros_like(sample_step_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, sample_step_tensor)
            sample_step = sample_step_max = max(t.item() for t in gathered)
            logger.info(f"Rank {rank}: ODE sample_step={orig_sample_step} (t_raw={t_raw:.1f}, matched_t={timesteps_candidates[orig_sample_step]:.1f})")
        else:
            t1 = getattr(args, "refl_timestep_t1", None)
            t2 = getattr(args, "refl_timestep_t2", None)

            num_steps = args.refl_sampling_steps
            t1 = max(0, min(t1, num_steps - 1))
            t2 = max(t1, min(t2, num_steps - 1))
            seed = self.ss.update_steps * 100000 + dp_rank
            gen = torch.Generator(device=device).manual_seed(seed)
            sample_step = torch.randint(t1, t2 + 1, (1,), device=device, generator=gen).item()
            # Sync sample_step across ranks to avoid deadlock (different ranks would run different # of denoising steps)
            # Gather all ranks' sample_step, take max, and use sample_step_max so all ranks run the same # of steps

            orig_sample_step = sample_step
            sample_step_tensor = torch.tensor([sample_step], dtype=torch.long, device=device)
            gathered = [torch.zeros_like(sample_step_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, sample_step_tensor)
            sample_step = sample_step_max = max(t.item() for t in gathered)
            logger.info(f"Rank {rank}: sample_step_max={sample_step_max} (gathered from all ranks, orig={orig_sample_step})")


        # ==================== Phase 1: Online Sample Generation ====================
        rollout_start_time = sync_cuda_time()
        all_latents, all_latents_audio, all_sample_inputs, all_channel_cond, all_save_paths, sigma_schedule = self.prepare_samples_online(
            model, batch, self.ss.update_steps, dp_rank, sp_rank, sp_group, sample_step
        )
        rollout_end_time = sync_cuda_time()
        rollout_time = rollout_end_time - rollout_start_time
        
        
        # ==================== Phase 2: Prepare Training Data ====================
        batch_size = all_latents.shape[0]
        
        # Prepare timesteps for all samples
        # Keep timesteps aligned with `all_latents` (Steps+1 states, indices 0..refl_sampling_steps):
        # the LRM reward path needs the "next state" timestep at orig_sample_step+1, which for the
        # last denoise step is the final (t~=0) state. Truncating to refl_sampling_steps dropped it
        # and made `samples["timesteps"][:, orig_sample_step + 1]` go out of bounds.
        timestep_value = [sigma * 1000 for sigma in sigma_schedule][:args.refl_sampling_steps + 1]
        timestep_values = [timestep_value[:] for _ in range(batch_size)]
        timesteps = torch.tensor(timestep_values, device=device, dtype=torch.float32)
        
        # Build samples dict with aligned latents and log_probs
        # Note: log_probs has length num_steps, latents has num_steps+1
        # We take latents[:, :-1] (pre-step) and latents[:, 1:] (post-step) to match log_probs/timesteps
        samples = {
            "timesteps": timesteps.detach().clone(),    # [batch, num_steps+1] (state timesteps, aligned with latents)
            "latents": all_latents,            
        }
        
        # Store the per-sample MoE sample-input contract + channel condition from all
        # samples (avoid recomputation during training). Kept as per-sample lists so the
        # shuffle below can reorder them and forward_step can stack them cfg-major.
        samples["sample_inputs"] = list(all_sample_inputs)                                        # list[dict], one per sample
        samples["channel_cond_vae_images"] = [cc.get("channel_cond_vae_images") for cc in all_channel_cond]  # list[tensor|None]

        if len(all_save_paths) == batch_size and len(prompts) == batch_size and len(seeds) == batch_size:
            samples["log_prompts"] = list(prompts)
            samples["log_seeds"] = [int(s.item()) if isinstance(s, torch.Tensor) else int(s) for s in seeds]
            samples["log_rollout_paths"] = list(all_save_paths)

        # Reuse the rollout audio latents for av runs: the JOINT av forward during the training-time
        # denoise needs the audio latent at the trained step as a (frozen) input, even though only the
        # visual reward is back-propagated. Audio has its own flow-shift, hence its own per-step timesteps;
        # build_audio_refl_scheduler is deterministic so it reproduces exactly the rollout audio schedule.
        if all_latents_audio is not None:
            samples["latents_audio"] = all_latents_audio                                          # [batch, Steps+1, ...]
            audio_scheduler_tmp = self.build_audio_refl_scheduler(device)
            audio_timestep_value = audio_scheduler_tmp.timesteps[:args.refl_sampling_steps].tolist()
            audio_timestep_values = [audio_timestep_value[:] for _ in range(batch_size)]
            samples["timesteps_audio"] = torch.tensor(audio_timestep_values, device=device, dtype=torch.float32)  # [batch, Steps]

        # ==================== Phase 4: ReFL Training Loop ====================
        mini_batch_size = len(samples["sample_inputs"])
        info = dict()

        refl_loop_start_time = sync_cuda_time()
        update_successful, grad_norm = self.forward_backward(
            model,
            samples,
            orig_sample_step,
            sp_rank,
            sigma_schedule,
            prompts,
            message_lists,
            mini_batch_size,
            info
        )
        
        refl_loop_end_time = sync_cuda_time()
        refl_loop_time = refl_loop_end_time - refl_loop_start_time

        # ==================== Phase 5: Aggregate Metrics and Return ====================
        # Aggregate training metrics across all timesteps
        # Use appropriate aggregation for different metric types
        info_aggregated = {}

        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                v_d = v.detach().clone()
                dist.all_reduce(v_d, op=dist.ReduceOp.AVG)
                info_aggregated[k] = v_d.item()
            else:
                info_aggregated[k] = v


        pass
        
        # Reduce timing metrics across ranks
        def reduce_time_metric(time_value):
            """Helper to reduce time metrics across ranks."""
            time_tensor = torch.tensor(time_value, device=device, dtype=torch.float32)
            dist.all_reduce(time_tensor, op=dist.ReduceOp.AVG)
            return time_tensor.item()
        
        rollout_time_avg = reduce_time_metric(rollout_time)
        refl_loop_time_avg = reduce_time_metric(refl_loop_time)

        dist.barrier()
        
        # Build return dictionary with all metrics
        # grad_norm is not equal between fsdp and megatron, fsdp grad_norm is 
        # the grad norm of 1/8 params, while megatron grad_norm is the grad norm
        # of all params across world_size dimensions
        return_dict = {
            "optimizer_step_successful": update_successful,
            "grad_norm": grad_norm,
            "rollout_time": rollout_time_avg,
            "refl_loop_time": refl_loop_time_avg
        }
        
        # Add aggregated training metrics
        return_dict.update(info_aggregated)
        
        # Add successes statistics for filter_middle_advantage monitoring
        if "successes" in samples:
            local_successes = samples["successes"]
            gathered_successes = gather_tensor(local_successes.to(device))
            total_samples = gathered_successes.numel()
            valid_samples = (gathered_successes == 1).sum().item()
            filtered_samples = (gathered_successes == 0).sum().item()
            return_dict["valid_samples_count"] = valid_samples
            return_dict["filtered_samples_count"] = filtered_samples
            return_dict["valid_samples_ratio"] = valid_samples / total_samples if total_samples > 0 else 1.0
        
        return return_dict
    
    def forward_backward(
        self,
        model,
        samples,
        orig_sample_step,
        sp_rank,
        sigma_schedule,
        prompts,
        message_lists,
        mini_batch_size,
        info
    ) -> Tuple[bool, float]:
        args = self.args
        self.logger.info(f'sp_rank {sp_rank}: calling forward_backward, update step: {self.ss.update_steps}, train_steps: {self.ss.train_steps}')
        self.model_engine.optimizer.zero_grad()
        grad_norm = 0.0
        with (
            profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)
            if getattr(args, 'profile', False) else contextlib.nullcontext()
        ) as prof:

            # one denoise step -> reward_inferencer -> reward -> loss
            loss = self.forward_step(
                samples, 
                prompts,
                orig_sample_step, 
                sigma_schedule, 
                message_lists,
                mini_batch_size,
                model,
                info,
                args.kl_weight,
                args.reference_model_offload,
                args.reward_config["fps"]
            )
            # Scale loss by gradient accumulation steps
            final_loss = loss / args.gradient_accumulation_steps
            final_loss.backward()

         
            if has_pinned_memory_pool(ACTIVATION_POOL_NAME):
                get_pinned_memory_pool(ACTIVATION_POOL_NAME).reset()

            # When mini_batch_size_per_update > 1, each inner loop processes multiple samples,
            # so train_steps should increase by mini_batch_size instead of 1
            self.ss.add(train_steps=mini_batch_size)

            # Perform optimizer step if we've accumulated enough gradients
            is_update_step = self.ss.train_steps % args.gradient_accumulation_steps == 0
            if is_update_step:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip_grad,
                    foreach=True,
                ).item()

                # In particular save_checkpoint() ends with cast_optimizer('cpu'), so
                # the update right after a checkpoint would otherwise crash with the optimizer state
                # on cpu and the grads on cuda. On-load here (cheap no-op when already on GPU).
                if getattr(self.model_engine, "optimizer_offloading", False):
                    self.model_engine.cast_optimizer('cuda')

                self.model_engine.optimizer.step()
                self.model_engine.lr_scheduler.step()

                self.ss.add(update_steps=1, current_run_update_steps=1)
                self.ss.lr = self.model_engine.optimizer.param_groups[0]["lr"]
                self.logger.info(f'Rank {self.rank}, sp_rank {sp_rank}: update step: {self.ss.update_steps}, train_steps: {self.ss.train_steps}, grad_norm: {grad_norm}')
        if getattr(args, 'profile', False):
            if self.rank == 0:
                with_sp = "_sp" if getattr(args, 'sequence_parallel', False) else ""
                file = f"{args.output_dir}/ptm_trace_rank{self.rank}_tp_{args.tensor_model_parallel_size}{with_sp}.json"
                prof.export_chrome_trace(file)
                self.logger.info(f"profiler trace saved to {file}")
                time.sleep(30)
            sys.exit(0)
        return True, grad_norm

    def _reward_norm_active(self):
        """Whether z-score normalization should be applied to the reward this step.

        Off entirely when ``use_reward_norm`` is False. When EMA is enabled we also
        wait until ``reward_ema_warmup_steps`` updates have accumulated, so the loss
        is not divided by an ill-estimated (possibly zero) std during warmup.
        """
        if not self.use_reward_norm:
            return False
        if self.use_reward_ema and self._reward_ema_updates < self.reward_ema_warmup_steps:
            return False
        return True

    def _reward_dim_stats(self, k):
        """Return (mean, std) for reward dimension ``k`` from the running stats."""
        mean_k = self.reward_ema_mean.get(k, 0.0)
        var_k = max(self.reward_ema_sq.get(k, 1.0) - mean_k * mean_k, 0.0)
        std_k = var_k ** 0.5
        return mean_k, std_k

    @torch.no_grad()
    def _update_reward_ema(self, reward):
        if not (self.use_reward_ema and isinstance(reward, dict)):
            return
        decay = self.reward_ema_decay
        # Sorted keys => identical collective order on every rank.
        for k in sorted(reward.keys()):
            r = reward[k].detach().float().reshape(-1)
            if self.sp_rank == 0:
                cnt = torch.tensor(float(r.numel()), device=r.device)
                s = r.sum()
                sq = (r * r).sum()
            else:
                cnt = torch.zeros((), device=r.device)
                s = torch.zeros((), device=r.device)
                sq = torch.zeros((), device=r.device)
            stats = torch.stack([cnt, s, sq])
            if dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            cnt_v, s_v, sq_v = stats[0].item(), stats[1].item(), stats[2].item()
            if cnt_v <= 0:
                continue
            batch_mean = s_v / cnt_v
            batch_meansq = sq_v / cnt_v
            if k not in self.reward_ema_mean:
                # Unseen dimension: initialize directly from the batch estimate.
                self.reward_ema_mean[k] = batch_mean
                self.reward_ema_sq[k] = batch_meansq
            else:
                self.reward_ema_mean[k] = decay * self.reward_ema_mean[k] + (1.0 - decay) * batch_mean
                self.reward_ema_sq[k] = decay * self.reward_ema_sq[k] + (1.0 - decay) * batch_meansq
        self._reward_ema_updates += 1

    def forward_step(
        self, 
        samples,
        prompts,
        orig_sample_step,
        sigma_schedule, 
        message_lists,
        mini_batch_size, 
        model,
        info,
        kl_weight,
        reference_model_offload,
        fps
    ):

        latents = samples["latents"][:, orig_sample_step]
        timesteps = samples["timesteps"][:, orig_sample_step]
        mmdit_kwargs = _stack_mini_inputs(samples["sample_inputs"])

        # channel_cond_vae_images is captured per-sample (NOT cfg-expanded) as [1, ...];
        # stack to [MB, ...]. None if the task has no channel condition (e.g. t2v).
        channel_cond_list = samples.get("channel_cond_vae_images")
        if channel_cond_list is not None and any(c is not None for c in channel_cond_list):
            channel_cond_vae_images = torch.cat(list(channel_cond_list), dim=0)
        else:
            channel_cond_vae_images = None

        # Reuse the rollout audio latents at the trained step for the JOINT av forward (None for
        # visual-only runs). Audio is a frozen input here -- only the visual reward is back-propagated.
        audio_latents = samples["latents_audio"][:, orig_sample_step] if "latents_audio" in samples else None
        audio_timesteps = samples["timesteps_audio"][:, orig_sample_step] if "timesteps_audio" in samples else None

        model_pred, scheduler, timesteps, pipeline, model_kwargs = self.refl_denoise_step(
            message_lists,
            model,
            latents,
            timesteps,
            sigma_schedule,
            channel_cond_vae_images=channel_cond_vae_images,
            audio_latents=audio_latents,
            audio_timesteps=audio_timesteps,
            **mmdit_kwargs,
        )

        device = model.device

        if kl_weight > 0:
            with torch.no_grad():
                if reference_model_offload:
                    self.logger.info(f'Rank {self.rank}: offloading model to CPU for KL computation')
                    self.ref_model.to(device)
                    torch.cuda.empty_cache()

                model_pred_ref, _, _, _, _ = self.refl_denoise_step(
                    message_lists,
                    self.ref_model,
                    latents,
                    timesteps,
                    sigma_schedule,
                    channel_cond_vae_images=channel_cond_vae_images,
                    audio_latents=audio_latents,
                    audio_timesteps=audio_timesteps,
                    **mmdit_kwargs,
                )

                if reference_model_offload:
                    self.logger.info(f'Rank {self.rank}: offloading ref_model to CPU after KL computation')
                    self.ref_model.to('cpu')
                    torch.cuda.empty_cache()

        # predict next step 
        single_timestep = timesteps if timesteps.dim() == 0 else timesteps[0]

        # If use LRM, x_t -> x_{t-1} -> reward
        # If not use LRM, x_t -> x_0 -> vae -> videos -> reward
        if self.args.reward_config["use_lrm"]:
            reward = self.refl_latent_reward(
                        samples, 
                        scheduler, 
                        pipeline, 
                        model_pred, 
                        orig_sample_step, 
                        single_timestep, 
                        latents, 
                        model_kwargs
                        )
        else:
            reward = self.refl_rgb_reward(
                        scheduler,
                        model_pred,
                        latents,
                        single_timestep,
                        prompts,
                        fps,
                        )
            

        # ReFL loss
        if isinstance(reward, torch.Tensor):
            loss_reward = torch.nn.functional.relu(-reward + self.reward_th).mean()
        elif isinstance(reward, dict):
    
            self._update_reward_ema(reward)
            norm_active = self._reward_norm_active()
            loss_reward = torch.tensor(0.0, device=device)
            for k, r in reward.items():
                # Per-dimension z-score normalization to equalize gradient scale
                if norm_active:
                    mean_k, std_k = self._reward_dim_stats(k)
                    r = (r - mean_k) / max(std_k, self.reward_norm_eps)
                loss_reward += self.args.reward_config["sub_reward"][k] * torch.nn.functional.relu(-r + self.reward_th).mean()
            loss_reward = loss_reward / len(reward)

        if kl_weight > 0:
            kl_loss = ((model_pred - model_pred_ref) ** 2).mean()
            loss = loss_reward + kl_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=device)
            loss = loss_reward

        # Information collection
        info["loss_reward"] = loss_reward
        info["kl_loss"] = kl_loss
        info["loss"] = loss
        if isinstance(reward, dict):
            for k, r in reward.items():
                info[f"gathered_hyvideoreward_remote_{k.lower()}_reward_mean"] = r
            # Expose the running normalization stats so drift/warmup can be watched.
            if self.use_reward_norm or self.use_reward_ema:
                for k in reward.keys():
                    mean_k, std_k = self._reward_dim_stats(k)
                    info[f"reward_norm_{k.lower()}_mean"] = mean_k
                    info[f"reward_norm_{k.lower()}_std"] = std_k
        else:
            info["reward"] = reward


        if self.sp_rank == 0:
            self._save_refl_rollout_records(reward, samples)

        return loss

    def refl_latent_reward(
                        self, 
                        samples, 
                        scheduler, 
                        pipeline, 
                        model_pred, 
                        orig_sample_step, 
                        single_timestep, 
                        latents, 
                        model_kwargs
                        ):

        prev_latents = scheduler.step(
                model_pred,
                single_timestep,
                latents,
                return_dict=False
            )[0]

        channel_cond_vae_images = model_kwargs.pop("channel_cond_vae_images")
        channel_cond_latents, channel_cond_mask, _ = pipeline.prepare_channel_cond_latents(
            channel_cond_vae_images,
            latents,
        )
        latent_model_input = torch.cat([prev_latents, channel_cond_latents, channel_cond_mask], dim=1)
        next_timestep = samples["timesteps"][:, orig_sample_step + 1]

        # get reward
        with torch.autocast("cuda", torch.bfloat16):
            reward = self.reward_inferencer(
                latents=latent_model_input,
                timesteps=next_timestep,
                **model_kwargs,
            )[0]
        
        return reward

    
    def refl_rgb_reward(
                        self,
                        scheduler,
                        model_pred,
                        latents,
                        single_timestep,
                        prompts,
                        fps,
                        ):

        clean_latents = scheduler.step_to_x0(
                model_pred,
                latents,
                single_timestep
            )
        
        # clean_latents = scheduler.step(
        #         model_pred,
        #         single_timestep,
        #         latents,
        #         return_dict=False
        #     )[0]

        # Denormalize latents from model space back to raw VAE space
        z = denormalize_vae_latents(self.vae, clean_latents.float())

        # vae decode with spatial tiling and checkpoint subtiles
        vae_autocast = getattr(self.args, "vae_autocast_dtype", "fp32")
        from hymm.utils.torch_utils import PRECISION_TO_TYPE
        dtype = PRECISION_TO_TYPE.get(vae_autocast, torch.float32)
        with torch.autocast(
            device_type="cuda", dtype=dtype, enabled=dtype != torch.float32,
        ):
            with eager_offload_context():
                videos = self.vae.decode(z, return_dict=False)[0]  # [1, 3, T', H', W']

        if self.sp_rank == 0:
            self._save_refl_rollout_videos(videos)


        if self.args.reward_config["reward_model"] == "qwen_rm":
            input_list, video_metas = self._prepare_qwen_rm_reward_inputs(videos, fps, prompts)
            reward = self.reward_inferencer.reward(input_list, prompts, video_metas=video_metas, mode="with_grad")
        elif self.args.reward_config["reward_model"] == "vjepa":
            raise NotImplementedError("VJepa reward model is not implemented yet")
        else:
            videos_input = (videos / 2 + 0.5).clamp(0, 1)
            reward = self.reward_inferencer.score_grad(prompts, videos_input)

        return reward

    def _prepare_qwen_rm_reward_inputs(self, videos, fps, prompts):
        """Build the ``(input_list, video_metas)`` fed to the Qwen-VL reward model.
        """
        sample_n_frames = videos.shape[2]
        rm_sample_n_frames = int(sample_n_frames // 24 * fps)

        if rm_sample_n_frames % 2 == 1:
            rm_sample_n_frames = rm_sample_n_frames - 1
        rm_sample_n_frames = max(rm_sample_n_frames, 2)

        if videos.shape[2] < rm_sample_n_frames:
            videos = videos.repeat(1, 1, rm_sample_n_frames - videos.shape[2], 1, 1)

        T = videos.shape[2]
        frame_indices = torch.linspace(0, T - 1, steps=rm_sample_n_frames, device=videos.device)
        frame_indices = torch.round(frame_indices).long()
        videos_input = videos.index_select(2, frame_indices)

        _, spatial_factor = self.reward_inferencer.patch_factors
        H, W = videos_input.shape[-2], videos_input.shape[-1]
        H_crop = H - (H % spatial_factor)
        W_crop = W - (W % spatial_factor)
        if (H_crop, W_crop) != (H, W):
            top, left = (H - H_crop) // 2, (W - W_crop) // 2
            videos_input = videos_input[..., top:top + H_crop, left:left + W_crop]

        videos_input = (videos_input / 2 + 0.5).clamp(0, 1) * 255.0

        input_list = [videos_input[i].transpose(0, 1) for i in range(videos_input.shape[0])]
        video_meta = {
            "frames_indices": frame_indices.tolist(),
            "fps": fps,
            "total_num_frames": T,
        }
        video_metas = [video_meta] * len(prompts)
        return input_list, video_metas

    def _save_refl_rollout_videos(self, videos):
        """Save the decoded x0 videos (the samples scored by the reward model) to disk.

        ``videos`` is the VAE-decoded x0 estimate of shape [B, C, T, H, W] in [-1, 1]
        (the exact tensor the reward is computed on). A .png is written per sample for
        single-frame (image) outputs and an .mp4 otherwise. Best-effort: detached from the
        autograd graph and any IO error is logged and swallowed so it never breaks the
        training step.
        """
        try:
            save_dir = os.path.join(self.args.output_dir, "rl_samples", f"{self.ss.train_steps:07d}")
            os.makedirs(save_dir, exist_ok=True)
            fps = getattr(self.args, "video_fps", 24)
            # Map to [0, 1] on a detached copy so the reward graph tensor is untouched.
            videos_save = (videos.detach().float() / 2 + 0.5).clamp(0, 1)
            num_samples = videos_save.shape[0]
            for b in range(num_samples):
                single = videos_save[b]  # [C, T, H, W]
                base = os.path.join(save_dir, f"dp_{self.dp_rank}_b_{b}_sample")
                if single.shape[1] == 1:
                    saved_path = base + ".png"
                    torchvision.utils.save_image(single[:, 0], saved_path)  # [C, H, W]
                else:
                    # [C, T, H, W] -> [T, H, W, C] uint8 for save_video_audio
                    frames = (single.permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype("uint8")
                    saved_path = base + ".mp4"
                    save_video_audio(frames, None, saved_path, fps=fps)
                self.logger.info(f"Rank {self.rank}: saved refl sample {b + 1}/{num_samples} to {saved_path}")
        except Exception as e:
            self.logger.warning(f"Rank {self.rank}: failed to save refl rollout videos: {e}")

    def _save_refl_rollout_records(self, reward, samples):
        """Dump per-sample rollout records to a JSON file in the current step's rl_samples folder.
        """
        try:
            save_dir = os.path.join(self.args.output_dir, "rl_samples", f"{self.ss.train_steps:07d}")
            os.makedirs(save_dir, exist_ok=True)

            prompts = samples.get("log_prompts")
            seeds = samples.get("log_seeds")
            rollout_paths = samples.get("log_rollout_paths")

            # Normalize reward into {dim: [B] float list} so each sample can be indexed.
            reward_dims = {}
            if isinstance(reward, dict):
                for k, r in reward.items():
                    reward_dims[k] = r.detach().float().reshape(-1).cpu().tolist()
            elif isinstance(reward, torch.Tensor):
                reward_dims["reward"] = reward.detach().float().reshape(-1).cpu().tolist()

            num_samples = max((len(v) for v in reward_dims.values()), default=0)
            if num_samples == 0 and prompts is not None:
                num_samples = len(prompts)

            records = []
            for b in range(num_samples):
                # Matches _save_refl_rollout_videos naming (.mp4 for video, .png for single frame).
                video_path = os.path.join(save_dir, f"dp_{self.dp_rank}_b_{b}_sample.mp4")
                records.append({
                    "sample_idx": b,
                    "video_path": video_path,
                    "prompt": prompts[b] if prompts is not None and b < len(prompts) else None,
                    "seed": seeds[b] if seeds is not None and b < len(seeds) else None,
                    "rollout_path": rollout_paths[b] if rollout_paths is not None and b < len(rollout_paths) else None,
                    "reward": {k: (v[b] if b < len(v) else None) for k, v in reward_dims.items()},
                })

            out_path = os.path.join(save_dir, f"rollout_records_dp_{self.dp_rank}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "train_steps": int(self.ss.train_steps),
                    "update_steps": int(self.ss.update_steps),
                    "dp_rank": self.dp_rank,
                    "records": records,
                }, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Rank {self.rank}: saved {len(records)} rollout records to {out_path}")
        except Exception as e:
            self.logger.warning(f"Rank {self.rank}: failed to save refl rollout records: {e}")


    def train(self):
        args = self.args
        device = torch.cuda.current_device()

        rank = self.rank
        local_rank = self.local_rank
        world_size = self.world_size
        parallel_dims = getattr(self, 'parallel_dims', None) # if parallel_dims is none, hy_parallelism.parallel_states.get_parallel_state is used
        dp_degree = self.dp_degree
        dp_rank = self.dp_rank
        sp_rank = self.sp_rank
        sp_group = self.sp_group
        sp_size = self.sp_size
        all_rank_logger.info(f"local_rank: {local_rank}, world_size: {world_size}, dp_degree: {dp_degree}, dp_rank: {dp_rank}, sp_rank: {sp_rank}, sp_size: {sp_size}")

        logger = self.logger
        model = self.model
        ref_model = self.ref_model


        # ============================== Build Env ==============================
        if has_pinned_memory_pool(ACTIVATION_POOL_NAME):
            get_pinned_memory_pool(ACTIVATION_POOL_NAME).reset()

        # ============================== Build Video Dataset ==============================
        video_dataset = self.video_dataset
        video_sampler = self.video_sampler
        video_loader = self.video_loader
        logger.info(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")

                
        # ============================== Print Key Info. ==============================
        video_total_batch_size = args.video_micro_batch_size[-1] * (world_size // dp_degree) if video_dataset is not None else 0

        video_num = 0
        if video_dataset is not None:
            video_num = video_dataset.total_length

        try:
            print_training_configuration(args, logger, model, world_size, local_rank, rank, 
                                    dp_degree, dp_rank, sp_size, sp_rank, video_total_batch_size, video_num,
                                    video_loader, video_dataset, self.ss)
        except Exception as e:
            # don't break training for any error in printing training configuration
            logger.warning(f"Error in print_training_configuration: {e}")
        
        log_gpu_memory_usage("before training, before aggressive empty cache", logger=logger)
        aggressive_empty_cache(force_sync=True)
        log_gpu_memory_usage("before training, after aggressive empty cache", logger=logger)

        # ============================= Resume data position =============================
        # Continue from the epoch stored in the (base-restored) scalar state, and skip the
        # data batches already consumed within that epoch for exact data-position resume.
        is_resuming = bool(args.resume and args.resume != "None")
        start_epoch = self.ss.epoch if is_resuming else 0
        resumed_epoch_consumed_batches = getattr(self, "_resumed_epoch_consumed_batches", 0) if is_resuming else 0
        # start_index (unique dataset items) = consumed_batches * total_batch_size is only
        # a valid mapping when repeat_count == 1 (the current data pipeline setting).
        resumed_start_index = resumed_epoch_consumed_batches * getattr(video_sampler, "total_batch_size", 1)
        can_skip_data = (
            resumed_epoch_consumed_batches > 0
            and getattr(video_sampler, "repeat_count", 1) == 1
            and video_dataset is not None
            and resumed_start_index < len(video_dataset)
        )
        if resumed_epoch_consumed_batches > 0 and not can_skip_data:
            logger.warning(
                f"[Resume] cannot exactly skip {resumed_epoch_consumed_batches} consumed batches "
                f"(repeat_count={getattr(video_sampler, 'repeat_count', None)}, "
                f"start_index={resumed_start_index}, dataset_len={len(video_dataset) if video_dataset is not None else None}). "
                "Data will restart from the beginning of the resumed epoch."
            )

        # ============================= Start training =============================
        for epoch in range(start_epoch, args.max_epochs):
            video_sampler.set_epoch(epoch)
            # Set the data start position for this epoch. Only the first resumed epoch
            # skips already-consumed batches; subsequent epochs start from 0.
            if is_resuming and epoch == start_epoch and can_skip_data:
                self._epoch_consumed_batches = resumed_epoch_consumed_batches
                video_sampler.start_index = resumed_epoch_consumed_batches * video_sampler.total_batch_size
                logger.info(
                    f"[Resume] epoch {epoch}: skipping {resumed_epoch_consumed_batches} batches "
                    f"(start_index={video_sampler.start_index}, remaining len={len(video_sampler)})."
                )
            else:
                self._epoch_consumed_batches = 0
                if getattr(video_sampler, "start_index", 0) != 0:
                    video_sampler.start_index = 0

            data_iter = iter(video_loader)
            global_start_time = sync_cuda_time()

            while True:
                # =========================== Data Loading =============================
                try:
                    batch = next(data_iter)
                    batch = sync_object_for_parallel_training(batch, parallel_dims=parallel_dims) # syncing is cheap: <0.01s
                except StopIteration:
                    break

                # for exact batch-level resume
                self._epoch_consumed_batches += 1

                sync_random_states(parallel_dims)
                self.ss = sync_object_for_parallel_training(self.ss, parallel_dims=parallel_dims, force_object=True) # syncing is cheap: < 0.01s
                
                # =========================== Sample Validation =============================
                logger.info(f"Rank {dp_rank}: sample validation at update_steps {self.ss.update_steps}, sample_interval {args.sample_interval} {self.ss.update_steps % args.sample_interval == 0}")
                if (args.sample_interval > 0 and (self.ss.update_steps > 0 and (self.ss.update_steps % args.sample_interval == 0))) or getattr(args, 'eval_first_iter', False):
                    logger.info(f"Rank {dp_rank}: sample validation at update_steps {self.ss.update_steps}")
                    self.sample_validation()
                    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

                # =========================== Training Loop =============================
                loss_dict = self.train_one_step(
                                model, 
                                ref_model, 
                                self.reward_inferencer, 
                                sp_rank, 
                                sp_group, 
                                sp_size, 
                                dp_degree,
                                batch, 
                                device, 
                                dp_rank
                            )
                global_start_time = self.report_training_progress(global_start_time, loss_dict)

                # =========================== Checkpointing =============================
                logger.info(f"--> ss.update_steps {self.ss.update_steps}, args.checkpointing_steps {args.checkpointing_steps}")
                if self.ss.update_steps % args.checkpointing_steps == 0 and self.ss.update_steps > 0:
                    logger.info(f"--> save checkpoint at step {self.ss.update_steps}, {args.output_dir}")
                    self.save_checkpoint()
                    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

            new_epoch = self.ss.inc_epoch()
            logger.info(f"Increase epoch to {new_epoch}.")

        self.save_checkpoint()

    
    def report_training_progress(self, global_start_time, loss_dict):
        args = self.args
        #################### LOGGING ####################
        is_update_step = self.ss.train_steps % args.gradient_accumulation_steps == 0
        if is_update_step and self.ss.update_steps % args.log_interval == 0:
            # Measure training speed:
            end_time = sync_cuda_time()
            steps_this_window = args.log_interval
            sec_per_step = (end_time - global_start_time) / steps_this_window
            steps_per_sec = steps_this_window / (end_time - global_start_time)
            global_start_time = sync_cuda_time()
        if is_update_step and self.ss.update_steps % args.log_interval == 0:
            # Simplified progress info - only log loss_dict contents
            progress_info = {
                "epoch": f"{self.ss.epoch}/{args.max_epochs}",
                "step": f"{self.ss.update_steps}",
                "learning_rate": f"{self.get_last_lr():.8f}",
                "step_time": f"{sec_per_step:.2f}s",
                "steps_per_sec_time": f"{steps_per_sec:.2f}",
            }

            # Add all loss_dict contents
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    progress_info[key] = f"{value.item():.8f}"
                else:
                    progress_info[key] = f"{value:.8f}"

            # logger is created on rank=0 while monitor is created on rank=world_size-1
            self.logger.info(f"Training Progress: update_steps {self.ss.update_steps}\ttrain_steps {self.ss.train_steps}\t{progress_info}")

            if self.rank == self.world_size-1:
                summary_events = []
                # Simplified tensorboard logging - only log loss_dict contents and basic metrics
                # Log all loss_dict contents
                for key, value in loss_dict.items():
                    scalar_value = value.item() if isinstance(value, torch.Tensor) else value
                    if 'advantages' in key:
                        # Log advantages statistics to dedicated Advantages group
                        summary_events.append((f"Advantages/{key}", scalar_value, self.ss.update_steps))
                    elif 'reward' in key:
                        summary_events.append((f"Reward/{key}", scalar_value, self.ss.update_steps))
                    elif 'time' in key:
                        summary_events.append((f"Time/{key}", scalar_value, self.ss.update_steps))
                    elif 'ratio_mean_t' in key or 'ratio_std_t' in key or 'clipfrac_t' in key or 'clipfrac_gt_one_t' in key or 'clipfrac_lt_one_t' in key:
                        # 按时间步分组的 ratio 统计信息
                        # 提取时间步索引，例如 ratio_mean_t5 -> t5
                        match = re.search(r'_t(\d+)$', key)
                        if match:
                            timestep = match.group(1)
                            if 'ratio_mean' in key: metric_type = 'mean'
                            elif 'ratio_std' in key: metric_type = 'std'
                            elif 'clipfrac_gt_one' in key: metric_type = 'clipfrac_gt_one'
                            elif 'clipfrac_lt_one' in key: metric_type = 'clipfrac_lt_one'
                            elif 'clipfrac' in key: metric_type = 'clipfrac'
                            else: metric_type = 'unknown'
                            
                            summary_events.append((f"RatioByTimestep/{metric_type}/t{timestep}", scalar_value, self.ss.update_steps))
                    elif 'ratio' in key:
                        summary_events.append((f"Ratio/{key}", scalar_value, self.ss.update_steps))
                    else:
                        summary_events.append((f"Train/{key}", scalar_value, self.ss.update_steps))

                # Log learning rate
                summary_events.append(("Train/learning_rate", self.get_last_lr(), self.ss.update_steps))
                # Log basic timing metrics
                summary_events.append(("Time/step_time", sec_per_step, self.ss.update_steps))
                summary_events.append(("Time/steps_per_sec", steps_per_sec, self.ss.update_steps))
                self.write_events(summary_events)

        return global_start_time

    def write_events(self, summary_events):
        self.model_engine.monitor.write_events(summary_events)

    def get_last_lr(self) -> float:
        return self.model_engine.lr_scheduler.get_last_lr()[0]


    def _audio_flow_shift(self):
        """Audio uses its own flow-shift (independent of the visual one)."""
        args = self.args
        shift = getattr(args, "flow_shift_audio", None)
        if shift is None:
            shift = getattr(getattr(self.model, "generation_config", None), "flow_shift_audio", None)
        if shift is None:
            shift = args.infer_flow_shift_video
        return shift

    def build_audio_refl_scheduler(self, device):
        """Build the audio ReFL scheduler with the audio-specific flow-shift.

        ``set_timesteps`` is deterministic given (shift, num_inference_steps), so this reproduces exactly the
        audio noise schedule used by the rollout pipeline's audio scheduler. The training-time joint denoise
        reuses the rollout audio latents at the matching audio timestep (see samples["timesteps_audio"]).
        """
        scheduler = FlowMatchDiscreteScheduler(
            shift=self._audio_flow_shift(),
            reverse=True,
            solver="euler",
        )
        scheduler.set_timesteps(num_inference_steps=self.args.refl_sampling_steps, device=device)
        scheduler._step_index = None
        scheduler._begin_index = None
        return scheduler

    def create_pipeline(self, model, scheduler):
        # Audio scheduler: built whenever an audio VAE is present, because for av tasks the rollout always
        # runs the audio diffusion branch. Audio has its own flow-shift, so its noise schedule differs from
        # the visual one; the training-time joint denoise reuses a matching audio schedule + audio timesteps
        # (see build_audio_refl_scheduler / samples["timesteps_audio"]). The audio VAE / processor are passed
        # so the rollout can decode audio by default.
        audio_scheduler = None
        audio_vae = model.model_dict.get("audio_vae") if hasattr(model, "model_dict") else None
        audio_processor = getattr(model, "audio_processor", None)
        if getattr(self.args, "use_audio_vae", False):
            audio_scheduler = FlowMatchDiscreteScheduler(
                shift=self._audio_flow_shift(),
                reverse=True,
                solver="euler",
            )

        # TODO pipeline correctness verification
        return Leo2ReFLPipeline(
            model=model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            scheduler=scheduler,
            vae_autocast_dtype=PRECISION_TO_TYPE[self.args.vae_autocast_dtype],
            args=self.args,
            video_scheduler=scheduler,
            audio_scheduler=audio_scheduler,
            audio_vae=audio_vae,
            audio_processor=audio_processor,
        )

    def rollout_pipeline(
        self,
        pipeline,
        model,
        batch_prompts,
        batch_message_lists,
        batch_seeds,
        target_height,
        target_width,
        target_length,
        generator_arg,
        determistic=None,
        sample_step=None
    ):
        args = self.args
        # Configure classifier-free guidance
        # Use rollout_cfg_scale for rollout function
        guidance_scale = getattr(args, "rollout_cfg_scale", 6.0)
        do_classifier_free_guidance = guidance_scale > 1.0
        model.generation_config.diff_guidance_scale = guidance_scale # align rollout_cfg_scale and diff_guidance_scale
        self.logger.info(f'Rank {self.rank}, rollout_phase: guidance_scale: {guidance_scale}, do_classifier_free_guidance: {do_classifier_free_guidance}')


        # leo2 pipeline
        # TODO It's better to reuse LeoModelHF.generate for rollout if the followings are done:
        # - LeoModelHF supports GRPO pipleline keywargs such as eta, sde_type
        # - LeoModelHF.generate returns all intermediate latents, log_probs, cond_text_states, cond_text_mask, 
        #  attention_mask, rope_media_info needed for GRPO training, instead of just the final video
        if self.model.generation_config.bot_task == "image":
            prepare_model_input_kwargs = dict(
                mode="gen_image",
                media_size=[target_height, target_width],
            )
        elif self.model.generation_config.bot_task == "video":
            prepare_model_input_kwargs = dict(
                mode="gen_video",
                video_fps=args.video_fps,
                ref_mode=self.model.generation_config.ref_mode,
                media_size=f"{target_height}x{target_width}",
            )
        elif self.model.generation_config.bot_task == "av":
            prepare_model_input_kwargs = dict(
                mode="gen_av",
                video_fps=args.video_fps,
                ref_mode=self.model.generation_config.ref_mode,
                media_size=f"{target_height}x{target_width}",
            )
        else:
            raise ValueError(f"Unsupported bot_task {self.model.generation_config.bot_task} for Leo2 pipeline")
        # Prepare system prompt
        # leo 2 pipeline
        num_frames = args.t2vi2v_task_kwargs.get("num_frames", args.num_frames)
        self.logger.info(f"prepare_model_inputs - message_list(len={len(batch_message_lists)})[0]: {batch_message_lists[0]}, media_size: {[target_height, target_width]}, num_frames: {num_frames}, {prepare_model_input_kwargs}")
        model_inputs = model.prepare_model_inputs(
            prompt=None, image=None, message_list=batch_message_lists, use_system_prompt=self.model.generation_config.use_system_prompt,
            seed=batch_seeds,
            num_frames=num_frames, **prepare_model_input_kwargs
        )
        
        _generator = model_inputs.pop("generator", None)
        model_inputs_reuse = deepcopy(model_inputs)
        if _generator is not None:
            model_inputs["generator"] = _generator

        if self.model.generation_config.bot_task == "image":
            batch_gen_image_info = model_inputs["batch_gen_image_info"]
            image_size = [batch_gen_image_info[0].image_height, batch_gen_image_info[0].image_width]
            self.logger.info(f"batch_gen_image_info: {model_inputs['batch_gen_image_info']}")
        elif self.model.generation_config.bot_task == "video":
            batch_gen_video_info = model_inputs["batch_gen_video_info"]
            image_size = [batch_gen_video_info[0].video_height, batch_gen_video_info[0].video_width]
            self.logger.info(f"batch_gen_video_info: {model_inputs['batch_gen_video_info']}")
        elif self.model.generation_config.bot_task == "av":
            batch_gen_video_info = model_inputs["batch_gen_video_info"]
            image_size = [batch_gen_video_info[0].video_height, batch_gen_video_info[0].video_width]
            self.logger.info(f"batch_gen_video_info: {model_inputs['batch_gen_video_info']}")
            self.logger.info(f"batch_gen_audio_info: {model_inputs['batch_gen_audio_info']}")

        # Whether audio participates in the forward / rollout is controlled by use_audio_vae: when on, the
        # model is treated as a joint audio+video generator, so the audio branch must run (and audio is
        # decoded by default). This is independent of grpo_use_audio, which only controls the GRPO objective
        # (see compute_log_prob).
        is_av = getattr(args, "use_audio_vae", False)
        pipeline_kwargs = {
            "batch_size": len(batch_prompts),
            "image_size": image_size,
            "video_duration": target_length,
            # Must match the rest of the ReFL pipeline (prepare_samples_online sets the visual scheduler and
            # build_audio_refl_scheduler both to refl_sampling_steps); otherwise the stored sigma_schedule /
            # audio timesteps would not line up with the latents actually produced during rollout.
            "num_inference_steps": args.refl_sampling_steps,
            "guidance_scale": guidance_scale,
            "generator": generator_arg,
            # For av, decode audio by default during rollout (the pipeline only decodes when an audio VAE is
            # present and the output_type carries an 'audio' entry).
            "output_type": dict(visual="pt", audio="pt") if is_av else "pt",
            "sde_type": args.sde_type,
            "return_dict": True,
            "eta": args.eta,
            "model_kwargs": model_inputs,
            "sample_step": sample_step,
        }
        if is_av:
            pipeline_kwargs["audio_duration"] = model_inputs["batch_gen_audio_info"][0].audio_duration

        # if output_type is np, return np shape is (B, T/F, H, W, C)
        # if output_type is pt, return tensor shape is (B, T/F, C, H, W)
        # Add determistic parameter (always pass it, whether it's a list, False, or True)
        if determistic is not None:
            pipeline_kwargs["determistic"] = determistic

        # Leo2ReFLPipeline.__call__ returns (all_latents, all_latents_audio); the audio track is
        # None for visual-only runs and the rollout audio latents [B, Steps+1, ...] for av runs.
        batch_latents_batch, batch_latents_audio_batch = pipeline(**pipeline_kwargs)
        # --------------------------------------------------------------------
        # 5.4: Pre-compute Prompt Embeddings for Training
        # --------------------------------------------------------------------
        # Pre-compute embeddings to avoid recomputation during training loop
        # Reuse the clean deepcopy captured before the rollout pipeline mutated model_inputs.
        model_inputs = model_inputs_reuse
        model_inputs = pipeline.encode_prompt(model_inputs)
        attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
            model_inputs['input_ids'], self.model.generation_config, model_kwargs=model_inputs,
        )
        model_inputs["attention_mask"] = attention_mask.to(model_inputs['input_ids'].device).to(dtype=torch.long)
        
        # NOTE: When do_classifier_free_guidance=True, prepare_model_inputs uses
        # cfg_factor=2, and apply_chat_template/batch_gen_infer arranges the output as
        # [cond_0, cond_1, ..., cond_{B-1}, uncond_0, uncond_1, ..., uncond_{B-1}]
        # (concatenated, NOT interleaved). The same layout applies to cond_text_states,
        # cond_text_mask, attention_mask and rope_media_info (rope_media_info is built
        # via `... * cfg_factor` list-concat in leo_hf.prepare_model_inputs).
        # So for sample i, the right (cond, uncond) pair lives at indices [i, i+B].
        B = len(batch_prompts)
        batch_sample_inputs = []
        batch_condition_embeds_list = []
        for i in range(B):
            if do_classifier_free_guidance:
                idx = [i, i + B]
            else:
                idx = [i]

            batch_sample_inputs.append(_capture_sample_inputs(model_inputs, idx))

            batch_condition_embeds_list.append({
                # channel condition, w/o cfg expand, only works for i2v/fi2v
                "channel_cond_vae_images": model_inputs['channel_cond_vae_images'][i:i+1].detach().clone() if model_inputs.get('channel_cond_vae_images') is not None else None,
            })

        sample0 = batch_sample_inputs[0]
        self.logger.info(
            f"batch_sample_inputs.len={len(batch_sample_inputs)}, "
            f"cond_text_states.shape={tuple(sample0['cond_text_states'].shape)}, "
            f"cond_text_mask.shape={tuple(sample0['cond_text_mask'].shape) if sample0['cond_text_mask'] is not None else None}, "
            f"attention_mask.shape={tuple(sample0['attention_mask'].shape) if sample0['attention_mask'] is not None else None}, "
            f"rope_media_info.len={len(sample0['rope_media_info']) if sample0['rope_media_info'] is not None else 0}, "
            f"rope_media_info={sample0['rope_media_info']}"
        )
        return batch_latents_batch, batch_latents_audio_batch, batch_sample_inputs, batch_condition_embeds_list

PARALLEL_GROUP_CACHE = {}

def get_parallel_groups(num_generations, sp_size, world_size):
    """
    Cache NCCL process groups so we do not recreate them every training step.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group must be initialized before creating subgroups.")
    cache_key = (num_generations, sp_size, world_size)
    cached = PARALLEL_GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dp_world_size = world_size // sp_size
    if dp_world_size % num_generations != 0:
        raise ValueError(
            f"dp_world_size ({dp_world_size}) must be divisible by num_generations ({num_generations})."
        )
    num_groups = dp_world_size // num_generations

    all_group_process_groups = []
    all_gather_groups = []
    all_group_leader_ranks = []

    # Keep creation order in sync across ranks
    dist.barrier()
    for g_idx in range(num_groups):
        g_dp_ranks = list(range(g_idx * num_generations, (g_idx + 1) * num_generations))
        g_all_global_ranks = []
        for dp_r in g_dp_ranks:
            g_all_global_ranks.extend([dp_r * sp_size + sp_r for sp_r in range(sp_size)])

        g_leader_rank = g_idx * num_generations * sp_size
        g_gather_ranks = [dp_r * sp_size for dp_r in g_dp_ranks]

        all_group_leader_ranks.append(g_leader_rank)
        all_group_process_groups.append(dist.new_group(ranks=g_all_global_ranks))
        all_gather_groups.append(dist.new_group(ranks=g_gather_ranks))

    PARALLEL_GROUP_CACHE[cache_key] = (
        all_group_process_groups,
        all_gather_groups,
        all_group_leader_ranks,
    )
    return PARALLEL_GROUP_CACHE[cache_key]


def sync_cuda_time(sync=True, barrier=True): # For accurate time measurement
    if barrier:
        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    if sync:
        torch.cuda.synchronize()
        t = time.time()
    else:
        t = time.time()
    return t
    
def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    # Ensure input tensor is contiguous for all_gather
    tensor = tensor.contiguous()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def batch_extra_kwargs(extra_kwargs_list):
    """
    Batch a list of per-sample extra_kwargs dicts into a single dict suitable for mini-batch processing.

    For each key:
      - If all values are tensors (and not None), stack them along batch dimension.
      - If all values are None, skip that key.
      - Otherwise, keep the list of values as-is.
    """
    if extra_kwargs_list is None:
        return {}

    # Ensure we are working with a list
    if not isinstance(extra_kwargs_list, (list, tuple)):
        return extra_kwargs_list

    if len(extra_kwargs_list) == 0:
        return {}

    # If elements are not dicts, return as-is (fallback)
    if not isinstance(extra_kwargs_list[0], dict):
        return extra_kwargs_list

    batched = {}
    # Collect all keys across dicts
    all_keys = set().union(*(d.keys() for d in extra_kwargs_list if isinstance(d, dict)))

    for key in all_keys:
        vals = []
        for d in extra_kwargs_list:
            if isinstance(d, dict):
                vals.append(d.get(key, None))
            else:
                vals.append(None)

        # All None -> skip
        if all(v is None for v in vals):
            continue

        # If all are tensors, stack into a batch
        if all(isinstance(v, torch.Tensor) for v in vals):
            batched[key] = torch.stack(vals, dim=0)
        else:
            # Mixed types: keep as list for safety
            batched[key] = vals

    return batched

def sync_random_tensor(
    generator_fn,
    shape,
    dtype=torch.long,
    device=None,
    sp_group=None,
):
    """
    Generate a random tensor and sync across SP group if needed.
    
    IMPORTANT: All SP ranks MUST call generator_fn() to keep global random state in sync!
    Only SP rank 0's result is used, but all ranks must advance their random state.
    
    Args:
        generator_fn: Function that generates the random tensor (e.g., lambda: torch.randperm(n))
        shape: Shape tuple for the tensor (e.g., (n,) for 1D, (m, n) for 2D)
        dtype: Data type of the tensor
        device: Device for the tensor
        sp_group: Sequence parallel group
    
    Returns:
        Synchronized random tensor
    """
    # CRITICAL: All ranks must call generator_fn() to keep global random state synchronized
    # Otherwise, subsequent random operations (e.g., randn_tensor in SDE sampling) will diverge
    tensor = generator_fn()
    
    if sp_group is not None:
        # Broadcast SP rank 0's tensor to all other SP ranks
        sp_leader_rank = dist.get_process_group_ranks(sp_group)[0]
        dist.broadcast(tensor, src=sp_leader_rank, group=sp_group)
    
    return tensor

def _normalize_group(
    rewards: torch.Tensor,
    ranks_per_group: int = 1,
    num_groups: int = 1,
    samples_per_grp: Optional[int] = None
) -> torch.Tensor:
    """
    Normalize rewards to compute advantages, with support for both single-rank and multi-rank groups.
    
    Args:
        rewards: Reward tensor to normalize
        ranks_per_group: Number of ranks per group (1 for single-rank, >1 for multi-rank)
        num_groups: Number of groups per rank (only used when ranks_per_group == 1)
        samples_per_grp: Number of samples per group (only used when ranks_per_group == 1)
    
    Returns:
        Normalized advantages tensor with same shape as rewards
    """
    advantages = torch.full_like(rewards, float("nan"))
    
    def _normalize_values(values: torch.Tensor) -> torch.Tensor:
        """Helper function to normalize a tensor with support for 1D and 2D tensors."""
        result = torch.full_like(values, float("nan"))
        finite_mask = torch.isfinite(values)
        
        if not finite_mask.any():
            return result
        
        if values.ndim == 1:
            # 1D case: simple normalization across all values
            valid = values[finite_mask]
            mean = valid.mean()
            std = torch.clamp(valid.std(unbiased=False), min=1e-6)
            result[finite_mask] = (valid - mean) / std
        elif values.ndim == 2:
            # 2D case: normalize along axis 0 (across samples), keep timestep dimension
            # Shape: [batch_size, num_timesteps]
            # mean/std shape: [1, num_timesteps]
            mean = values.mean(dim=0, keepdim=True)  # Mean across samples for each timestep
            std = torch.clamp(values.std(dim=0, keepdim=True, unbiased=False), min=1e-6)  # Std across samples for each timestep
            result = (values - mean) / std
            result[~finite_mask] = float("nan")
        else:
            # Fallback for other dimensions
            valid = values[finite_mask]
            mean = valid.mean()
            std = torch.clamp(valid.std(unbiased=False), min=1e-6)
            result[finite_mask] = (valid - mean) / std
        
        return result
    
    if ranks_per_group == 1:
        # Single-rank groups: normalize each group separately
        if samples_per_grp is None:
            samples_per_grp = len(rewards) // num_groups if num_groups > 0 else len(rewards)
        
        for grp_idx in range(num_groups):
            start_idx = grp_idx * samples_per_grp
            end_idx = start_idx + samples_per_grp
            slice_obj = slice(start_idx, min(end_idx, len(rewards)))
            if slice_obj.start >= slice_obj.stop:
                continue
            advantages[slice_obj] = _normalize_values(rewards[slice_obj])
    else:
        # Multi-rank groups: rewards already gathered for the full group, normalize all together
        advantages = _normalize_values(rewards)
    
    return advantages

def compute_weighted_advantages(
    samples: dict,
    reward_config: dict,
    ranks_per_group: int,
    num_groups: int,
    samples_per_grp: int,
    logger,
) -> torch.Tensor:
    """
    Compute weighted advantages using Method 2: Separate Advantage First.
    
    Correct formula:
        A_total = w1 * Normalize(R1) + w2 * Normalize(R2)
    
    This approach first normalizes each reward separately to unit variance,
    then weights them. This ensures each reward contributes according to its
    weight regardless of its original scale.
    
    Args:
        samples: Dict containing reward tensors with keys like "{model}_{metric}_rewards"
        reward_config: Reward configuration dict with model weights and sub_reward weights
        ranks_per_group: Number of ranks per group for normalization
        num_groups: Number of groups per rank
        samples_per_grp: Number of samples per group
        logger: Logger instance
    
    Returns:
        Weighted advantage tensor: A_total = Σ(model_weight * Σ(metric_weight * Normalize(R_metric)))
    """
    # Parse reward_config to get weights
    if reward_config is None or "models" not in reward_config:
        # Fallback: use simple avg_rewards normalization if no config
        logger.warning("No valid reward_config found, falling back to simple avg normalization")
        rewards = samples.get("avg_rewards")
        if rewards is None:
            raise ValueError("samples must contain 'avg_rewards' when reward_config is not provided")
        return _normalize_group(rewards, ranks_per_group, num_groups, samples_per_grp)
    
    models_config = reward_config["models"]
    
    # Get reference shape from any reward tensor
    reference_shape = None
    for key in samples.keys():
        if key.endswith("_rewards") and key != "avg_rewards" and key != "ori_avg_rewards":
            reference_shape = samples[key].shape
            break
    
    if reference_shape is None:
        # Fallback to avg_rewards
        logger.warning("No individual reward metrics found, falling back to simple avg normalization")
        rewards = samples.get("avg_rewards")
        if rewards is None:
            raise ValueError("samples must contain 'avg_rewards'")
        return _normalize_group(rewards, ranks_per_group, num_groups, samples_per_grp)
    
    # Initialize weighted advantages tensor
    device = samples["avg_rewards"].device
    dtype = samples["avg_rewards"].dtype
    weighted_advantages = torch.zeros(reference_shape, device=device, dtype=dtype)
    total_weight = 0.0
    
    # Process each model in reward_config
    for model_name, model_config in models_config.items():
        if not isinstance(model_config, dict):
            continue
        
        model_weight = float(model_config.get("weight", 1.0))
        if model_weight == 0.0:
            continue
        
        sub_reward = model_config.get("sub_reward", None)
        
        if sub_reward is not None and isinstance(sub_reward, dict):
            # Model has sub-metrics (e.g., VQ, MQ, TA)
            for metric_name, metric_weight in sub_reward.items():
                metric_weight = float(metric_weight)
                if metric_weight == 0.0:
                    continue
                
                # Build the key: "{model_name}_{metric_name}_rewards" (lowercase)
                reward_key = f"{model_name}_{metric_name.lower()}_rewards"
                
                if reward_key not in samples:
                    # Try without lowercase
                    reward_key_alt = f"{model_name}_{metric_name}_rewards"
                    if reward_key_alt in samples:
                        reward_key = reward_key_alt
                    else:
                        logger.warning(f"Reward key '{reward_key}' not found in samples, skipping")
                        continue
                
                reward_tensor = samples[reward_key]
                
                # Normalize this reward separately
                normalized_advantage = _normalize_group(
                    reward_tensor, ranks_per_group, num_groups, samples_per_grp
                )
                normalized_advantage = torch.nan_to_num(normalized_advantage, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Weight and accumulate
                combined_weight = model_weight * metric_weight
                weighted_advantages = weighted_advantages + combined_weight * normalized_advantage
                total_weight += combined_weight
                
                logger.debug(f"Added advantage for {reward_key}: model_weight={model_weight}, metric_weight={metric_weight} reward_tensor={reward_tensor}, normalized_advantage={normalized_advantage}, weighted_advantages={weighted_advantages}, total_weight={total_weight}")
        else:
            # Model without sub-metrics (single score model like aesthetic, clipscore)
            reward_key = f"{model_name}_rewards"
            
            if reward_key not in samples:
                logger.warning(f"Reward key '{reward_key}' not found in samples, skipping")
                continue
            
            reward_tensor = samples[reward_key]
            
            # Normalize this reward separately
            normalized_advantage = _normalize_group(
                reward_tensor, ranks_per_group, num_groups, samples_per_grp
            )
            normalized_advantage = torch.nan_to_num(normalized_advantage, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Weight and accumulate
            weighted_advantages = weighted_advantages + model_weight * normalized_advantage
            total_weight += model_weight
            
            logger.debug(f"Added advantage for {reward_key}: model_weight={model_weight}, normalized_advantage={normalized_advantage}, weighted_advantages={weighted_advantages}, total_weight={total_weight}")
    
    # If no valid rewards were processed, fallback to avg_rewards
    if total_weight == 0.0:
        logger.warning("No valid rewards processed, falling back to simple avg normalization")
        rewards = samples.get("avg_rewards")
        if rewards is None:
            raise ValueError("samples must contain 'avg_rewards'")
        return _normalize_group(rewards, ranks_per_group, num_groups, samples_per_grp)
    
    logger.info(f"Computed weighted advantages with total_weight={total_weight:.4f} (Method 2: Separate Advantage First)")
    
    return weighted_advantages


def gather_and_process_rewards(
    reward_dicts,
    ranks_per_group,
    num_groups_per_rank,
    group_idx,
    rank_in_group,
    dp_rank,
    sp_rank,
    video_batch_size,
    args,
    device,
    global_step,
    indexs,
    sp_group,
    world_size,
    logger,
):
    """
    Simplified reward gathering logic for parallel groups mode.
    
    Returns:
        reward_scores: Dict of {metric_name: torch.Tensor} with gathered rewards
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
    # Get metric names dynamically from reward_dicts (returned by rewards.py)
    # reward_dicts contains keys like: "videoalign_local_vq", "aesthetic", "tencent_remote_mq", "avg"
    metric_names = set()
    if reward_dicts and len(reward_dicts) > 0:
        for rd in reward_dicts:
            # Exclude metadata keys and only keep reward score keys
            for key in rd.keys():
                if key not in ["prompt", "seed", "video_path", "sample_idx", "ranks_per_group", 
                               "group_idx_local", "global_group_idx", "sample_idx_in_group", 
                               "group_idx", "rank_in_group"]:
                    metric_names.add(key)
    
    # Convert to list, ensure 'avg' is included
    metric_names = list(metric_names)
    if 'avg' not in metric_names:
        metric_names.append('avg')
    
    # Step 1: Synchronize metric names across all ranks in the world
    # This ensures all ranks have the same metric names before any group-specific communication
    if torch.distributed.is_initialized() and world_size > 1:
        gathered_metric_names = [None] * world_size
        torch.distributed.all_gather_object(gathered_metric_names, metric_names)
        all_metric_names = set()
        for mn_list in gathered_metric_names:
            if isinstance(mn_list, list):
                all_metric_names.update(mn_list)
        all_metric_names.add('avg')
        metric_names = sorted(list(all_metric_names))
        logger.info(f"[GatherRewards] Rank {rank}: Synchronized metric names: {metric_names}")
    
    # Step 2: Setup communication groups
    group_process_group = None
    gather_group = None
    group_leader_rank = None
    broadcast_group = None
    broadcast_src_rank = None
    
    if ranks_per_group > 1:
        all_group_process_groups, all_gather_groups, all_group_leader_ranks = get_parallel_groups(
            ranks_per_group, sp_group.size(), world_size
        )
        group_process_group = all_group_process_groups[group_idx]
        gather_group = all_gather_groups[group_idx]
        group_leader_rank = all_group_leader_ranks[group_idx]
        broadcast_group = group_process_group
        broadcast_src_rank = group_leader_rank
        torch.cuda.empty_cache()
    elif sp_group is not None:
        # Single-rank groups: broadcast within sp_group
        broadcast_group = sp_group
        sp_group_ranks = torch.distributed.get_process_group_ranks(sp_group)
        broadcast_src_rank = sp_group_ranks[0]
    
    # Step 3: Collect rewards (only on sp_rank==0)
    reward_scores = {}
    if sp_rank == 0:
        # Extract local reward scores
        local_rewards = {}
        for metric in metric_names:
            scores = []
            for rd in reward_dicts:
                val = rd.get(metric, float('nan'))
                try:
                    scores.append(float(val) if val is not None else float('nan'))
                except (TypeError, ValueError):
                    scores.append(float('nan'))
            local_rewards[metric] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        if ranks_per_group > 1:
            # Multi-rank: gather rewards from all ranks in group
            for metric in metric_names:
                gathered = [torch.zeros_like(local_rewards[metric]) for _ in range(ranks_per_group)]
                torch.distributed.all_gather(gathered, local_rewards[metric], group=gather_group)
                reward_scores[metric] = torch.cat(gathered, dim=0)
            
            # Gather reward_dicts for saving
            gathered_reward_dicts = [None] * ranks_per_group
            torch.distributed.all_gather_object(gathered_reward_dicts, reward_dicts, group=gather_group)
            complete_reward_dicts = []
            for rd_list in gathered_reward_dicts:
                if isinstance(rd_list, list):
                    complete_reward_dicts.extend(rd_list)
            
            if rank_in_group == 0:
                save_dir = os.path.join(args.output_dir, "rl_samples", f"{global_step:07d}", f"group_{group_idx}")
                os.makedirs(save_dir, exist_ok=True)
                filename = f"index_{indexs[0] if isinstance(indexs, list) else indexs}_group_{group_idx}_rewards.json"
                with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
                    json.dump(convert_to_json_serializable(complete_reward_dicts), f, ensure_ascii=False, indent=4)
                logger.info(f"[GatherRewards] Rank {rank} saved {len(complete_reward_dicts)} entries for group {group_idx}")
        else:
            # Single-rank: use local rewards directly
            reward_scores = local_rewards
            
            # Save for each local group
            for grp_local_idx in range(num_groups_per_rank):
                global_grp_idx = dp_rank * num_groups_per_rank + grp_local_idx
                start_idx = grp_local_idx * args.num_generations
                end_idx = start_idx + args.num_generations
                group_dicts = reward_dicts[start_idx:end_idx]
                
                save_dir = os.path.join(args.output_dir, "rl_samples", f"{global_step:07d}", f"group_{global_grp_idx}")
                os.makedirs(save_dir, exist_ok=True)
                first_idx = indexs[start_idx] if start_idx < len(indexs) else 0
                filename = f"index_{first_idx}_group_{global_grp_idx}_rewards.json"
                with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
                    json.dump(convert_to_json_serializable(group_dicts), f, ensure_ascii=False, indent=4)
                logger.info(f"[GatherRewards] Rank {rank} saved {len(group_dicts)} entries for group {global_grp_idx}")
    else:
        # Non-sp_rank==0: initialize placeholder tensors
        expected_size = args.num_generations if ranks_per_group > 1 else video_batch_size
        for metric in metric_names:
            reward_scores[metric] = torch.zeros(expected_size, dtype=torch.float32, device=device)
    
    # Step 4: Broadcast rewards to all ranks (unified logic)
    if broadcast_group is not None:
        # Ensure all ranks have initialized reward_scores for all metrics
        expected_size = args.num_generations if ranks_per_group > 1 else video_batch_size
        for metric in metric_names:
            if metric not in reward_scores:
                reward_scores[metric] = torch.zeros(expected_size, dtype=torch.float32, device=device)
        
        # Broadcast all metrics
        for metric in metric_names:
            torch.distributed.broadcast(reward_scores[metric], src=broadcast_src_rank, group=broadcast_group)
    
    return reward_scores

def build_refl_scheduler(args, sigma_schedule, device):
    infer_flow_shift = args.infer_flow_shift_video
    scheduler = FlowMatchDiscreteScheduler(
        shift=infer_flow_shift,
        reverse=True,
        solver="euler",
    )
    sigma_schedule_local = sigma_schedule.to(device=device, dtype=torch.float32).clone()
    scheduler.num_inference_steps = sigma_schedule_local.shape[0] - 1
    scheduler.sigmas = sigma_schedule_local
    scheduler.timesteps = (sigma_schedule_local[:-1] * scheduler.config.num_train_timesteps).to(
        device=device, dtype=torch.float32
    )
    scheduler._step_index = None
    scheduler._begin_index = None
    return scheduler

def generate_visualization_html(step_dir, global_step, logger, upload_to_cos=True, media_height=240):
    """
    Generate visualization HTML for RL samples.
    Uses hymm.trainers.rl.utils.rl_visualization module.
    """
    try:
        from hymm.trainers.rl.utils.rl_visualization import generate_visualization_for_step
        generate_visualization_for_step(
            step_dir=step_dir,
            global_step=global_step,
            upload_to_cos=upload_to_cos,
            media_height=media_height,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Error in _generate_visualization_html: {e}")
        import traceback
        traceback.print_exc()

def print_training_configuration(args, logger, model, world_size, local_rank, rank, 
                                dp_degree, dp_rank, sp_size, sp_rank, video_total_batch_size, video_num,
                                video_loader=None, video_dataset=None, ss=None):
    """
    Print comprehensive training configuration information in a well-organized format.
    
    Args:
        args: Training arguments
        logger: Logger instance
        model: The model instance
        world_size: Total number of processes
        local_rank: Local rank within node
        rank: Global rank
        dp_degree: Data parallel degree
        dp_rank: Data parallel rank
        sp_size: Sequence parallel size
        total_batch_size: Total batch size across all GPUs
        image_num: Number of image samples
        video_num: Number of video samples
        image_loader: Image data loader (optional)
        video_loader: Video data loader (optional)
        image_dataset: Image dataset (optional)
        video_dataset: Video dataset (optional)
    """
    # params_count = model.params_count()
    
    logger.info("****************************** Running training ******************************")
    
    # ===============================================================================
    # System & Hardware Configuration
    # ===============================================================================
    logger.info("=" * 80)
    logger.info("SYSTEM & HARDWARE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Number of GPUs                                   : {world_size}")
    logger.info(f"Local rank                                       : {local_rank}")
    logger.info(f"Global rank                                      : {rank}")
    logger.info(f"Device                                           : cuda:{local_rank}")
    logger.info(f"Master weight dtype                              : {model.parameters().__next__().dtype}")
    
    # ===============================================================================
    # Model Parameters & Architecture
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("MODEL PARAMETERS & ARCHITECTURE")
    logger.info("-" * 80)
    # for k, v in params_count.items():
    #     logger.info(f"Number of {k:<25}                     : {v:,}")
    total_params_b = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    logger.info(f"Total trainable parameters per FSDP shard       : {total_params_b:.3f}B")
    logger.info(f"Model type                                       : {getattr(args, 'model_type', None)}")
    logger.info(f"Model name                                       : {getattr(args, 'model_name', None)}")
    logger.info(f"Model structure                                  : {getattr(args, 'structure', None)}")
    logger.info(f"MODEL_PATH                                       : {os.getenv('MODEL_PATH', 'None')}")
    if hasattr(args, 'text_encoder_2'):
        logger.info(f"Text encoder 2 type                              : {args.text_encoder_2}")
        logger.info(f"Text length 2                                    : {args.text_len_2}")
    if hasattr(args, 'recompute_granularity'):
        logger.info(f"Recompute granularity                            : {args.recompute_granularity}")
    if hasattr(args, 'recompute_num_layers'):
        logger.info(f"Recompute num layers                             : {args.recompute_num_layers}")
    if hasattr(args, 'attn_impl'):
        logger.info(f"Attention implementation                         : {args.attn_impl}")
        if hasattr(args, 'win_type'):
            logger.info(f"Window type                                      : {args.win_type}")
        if hasattr(args, 'win_size'):
            logger.info(f"Window size                                      : {args.win_size}")
    if hasattr(args, 'use_dynamic_ring_attention'):
        logger.info(f"Dynamic ring attention                           : {args.use_dynamic_ring_attention}")
    
    # ===============================================================================
    # Distributed Training Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("DISTRIBUTED TRAINING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"World size                                       : {world_size}")
    logger.info(f"DP degree                                        : {dp_degree}")
    logger.info(f"DP rank                                          : {dp_rank}")
    logger.info(f"SP size                                          : {sp_size}")
    logger.info(f"SP rank                                          : {sp_rank}")
    logger.info(f"CPU offload                                      : {args.use_cpu_offload}")
    
    # ===============================================================================
    # Batch Size & Data Flow Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("BATCH SIZE & DATA FLOW CONFIGURATION")
    logger.info("-" * 80)
    per_gpu_batch = args.micro_batch_size[0] if isinstance(args.micro_batch_size, list) else args.micro_batch_size
    dp_batch = per_gpu_batch * dp_degree
    effective_batch_video = video_total_batch_size * args.gradient_accumulation_steps
    
    logger.info(f"Micro batch size                                 : {args.micro_batch_size}")
    logger.info(f"Per GPU batch size                               : {per_gpu_batch}")
    logger.info(f"DP degree batch size                             : {dp_batch}")
    logger.info(f"Video micro batch size                           : {args.video_micro_batch_size}")
    logger.info(f"Video total batch size across all GPUs           : {video_total_batch_size}")
    logger.info(f"Gradient accumulation steps                      : {args.gradient_accumulation_steps}")
    
    # ===============================================================================
    # Dataset & DataLoader Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("DATASET & DATALOADER CONFIGURATION")
    logger.info("-" * 80)
    total_samples =  video_num
    logger.info(f"Data type                                        : {args.data_type}")
    logger.info(f"Total samples (image + video)                    : {total_samples:,}")
    logger.info(f"Video examples                                   : {video_num:,}")
    
    if hasattr(args, 'image_data_path') and 'image' in args.data_type:
        logger.info(f"Image data path                                  : {args.image_data_path}")
    if hasattr(args, 'video_data_path') and 'video' in args.data_type:
        logger.info(f"Video data path                                  : {args.video_data_path}")
    if hasattr(args, 'image_size'):
        logger.info(f"Image size                                       : {args.image_size}")
    if hasattr(args, 'video_bucket_hw_bucket_stride'):
        logger.info(f"Video bucket HW stride                           : {args.video_bucket_hw_bucket_stride}")
    
    if video_loader is not None:
        logger.info(f"Video dataloader batch size                      : {video_loader.batch_size}")
        logger.info(f"Video dataloader drop last                       : {video_loader.drop_last}")
        if video_dataset is not None:
            vid_len = len(video_dataset) if hasattr(video_dataset, '__len__') else 'Unknown'
            logger.info(f"Video dataset length                             : {vid_len}")
            if hasattr(video_dataset, 'total_length'):
                logger.info(f"Video dataset total length                       : {video_dataset.total_length}")
    
    # ===============================================================================
    # Training Schedule & Steps Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("TRAINING SCHEDULE & STEPS CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Number of epochs                                 : {args.num_train_epochs}")
    logger.info(f"Max training steps                               : {args.max_train_steps}")
    
    total_update_steps = args.max_train_steps // args.gradient_accumulation_steps
    logger.info(f"Total forward/backward steps                     : {args.max_train_steps}")
    logger.info(f"Total optimizer update steps                     : {total_update_steps}")
    logger.info(f"Accumulation steps per optimizer update          : {args.gradient_accumulation_steps}")
    
    # ===============================================================================
    # Optimization Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("OPTIMIZATION CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Optimizer                                        : {args.optimizer}")
    logger.info(f"Learning rate                                    : {args.lr}")
    logger.info(f"Weight decay                                     : {args.weight_decay}")
    logger.info(f"Max gradient norm                                : {args.clip_grad}")
    logger.info(f"LR scheduler                                     : {args.lr_scheduler}")
    logger.info(f"LR power                                         : {args.lr_power}")
    
    # ===============================================================================
    # Logging & Checkpointing Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("LOGGING & CHECKPOINTING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Output directory                                 : {args.output_dir}")
    logger.info(f"Checkpointing steps                              : {args.checkpointing_steps}")
    logger.info(f"Sample interval                                  : {args.sample_interval}")
    logger.info(f"Log interval                                     : {args.log_interval}")
    logger.info(f"Resume training                                  : {args.resume}")
    logger.info(f"Profiler enabled                                 : {args.profile}")
    logger.info(f"Dry run enabled                                  : {getattr(args, 'dry_run', False)}")
    
    # ===============================================================================
    # Inference & Validation Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("INFERENCE & VALIDATION CONFIGURATION")
    logger.info("-" * 80)
    if hasattr(args, 'infer_steps'):
        logger.info(f"Inference steps                                  : {args.infer_steps}")
    if hasattr(args, 'cfg_scale'):
        logger.info(f"CFG scale                                        : {args.cfg_scale}")
    if hasattr(args, 'infer_flow_shift'):
        logger.info(f"Inference flow shift                             : {args.infer_flow_shift}")
    if hasattr(args, 'neg_prompt'):
        logger.info(f"Negative prompt                                  : {args.neg_prompt}")
    if hasattr(args, 'vae_spatial_tiling'):
        logger.info(f"VAE spatial tiling                               : {args.vae_spatial_tiling}")
    if hasattr(args, 'vae_type'):
        logger.info(f"VAE version                                      : {args.vae_type}")
    
    # ===============================================================================
    # Seeds & Reproducibility Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("SEEDS & REPRODUCIBILITY CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Global seed                                      : {args.global_seed}")
    logger.info(f"Local seed                                       : {args.local_seed}")
    
    # ===============================================================================
    # Additional Configuration
    # ===============================================================================
    # 分拆逻辑：分别判断每个配置项，单独打印
    if hasattr(args, 'moe_loss_weight') and args.moe_loss_weight > 0:
        logger.info("-" * 80)
        logger.info("ADDITIONAL CONFIGURATION: MOE LOSS")
        logger.info("-" * 80)
        logger.info(f"MoE loss weight                                  : {args.moe_loss_weight}")

    if hasattr(args, 'convert_state_dict_in_out'):
        logger.info("-" * 80)
        logger.info("ADDITIONAL CONFIGURATION: CONVERT STATE DICT IN/OUT")
        logger.info("-" * 80)
        logger.info(f"Convert state dict in/out                        : {args.convert_state_dict_in_out}")

    # ===============================================================================
    # Performance Estimates
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("PERFORMANCE ESTIMATES")
    logger.info("-" * 80)
    
    # ===============================================================================
    # Mask Type Statistics
    # ===============================================================================
    if ss and getattr(ss, "consumed_samples_by_mask_type_total", None) is not None:
        logger.info("-" * 80)
        logger.info("MASK TYPE STATISTICS (RESUMED TRAINING)")
        logger.info("-" * 80)
        mask_type_stats = dict(ss.consumed_samples_by_mask_type_total)
        total_mask_samples = sum(mask_type_stats.values())
        for mt, count in mask_type_stats.items():
            percentage = (count / total_mask_samples * 100) if total_mask_samples > 0 else 0
            logger.info(f"Mask type '{mt}' samples                            : {count:,} ({percentage:.1f}%)")
        logger.info(f"Total mask type samples                          : {total_mask_samples:,}")
        
        # Data type statistics for t2v
        if ss.consumed_samples_by_data_type_total:
            logger.info("-" * 40)
            logger.info("DATA TYPE STATISTICS FOR T2V (RESUMED TRAINING)")
            logger.info("-" * 40)
            data_type_stats = dict(ss.consumed_samples_by_data_type_total)
            total_data_type_samples = sum(data_type_stats.values())
            for dt, count in data_type_stats.items():
                percentage = (count / total_data_type_samples * 100) if total_data_type_samples > 0 else 0
                logger.info(f"T2V Data type '{dt}' samples                        : {count:,} ({percentage:.1f}%)")
            logger.info(f"Total T2V data type samples                      : {total_data_type_samples:,}")
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)