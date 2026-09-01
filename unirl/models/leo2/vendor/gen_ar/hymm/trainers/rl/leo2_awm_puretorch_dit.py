
import sys
import os
import gc
import json
import re
import contextlib
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple, Type
import random
import numpy as np
import time
from functools import partial
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
from hymm.models.tokenizers import load_tokenizer
from hymm.core.global_vars import get_video_denoiser, get_audio_denoiser
from hymm.models.autoencoders import add_noise_to_latents, denormalize_vae_latents
from hymm.models.audio_encoders import add_audio_noise
from hymm.constants import SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES
from hymm.trainers.rl.utils.file_utils import validate_video_csv_files, convert_to_json_serializable
from hymm.utils.torch_utils import nanstd, PRECISION_TO_TYPE
from hymm.utils.helpers import default
from hymm.trainers.rl.utils.gc import (aggressive_empty_cache, log_gpu_memory_usage)
from hymm.ar.pipelines.pipeline_leo_rl import Leo2GRPOPipeline

from hymm.diffusion.schedulers import FlowMatchDiscreteScheduler
from hymm.trainers.helpers import GRPOTrainingStates
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

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


# --- Leo2.1 AWM per-sample input contract -------------------------------------
# CONDITIONAL leo.forward inputs (text / attention / RoPE / packing structure). Sliced from the pipeline
# model_inputs at rollout (conditional rows only, no CFG) by _slice_cond_sample_inputs. Do NOT add the flow
# tensors here: compute_log_prob's rollout branch builds latents / timesteps / ut / ... itself and splats
# these as **mmdit_kwargs, so a duplicate key would raise "multiple values for keyword argument".
_AWM_COND_TENSOR_KEYS = (
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
_AWM_COND_LIST_KEYS = (
    # attention / RoPE structure
    "rope_media_info",
)

# FLOW-matching forward inputs: the noised model inputs + velocity targets built by compute_log_prob's
# rollout branch, stored per sample, and replayed verbatim on the training recompute (NO re-noising):
# latents=v_x_t / timesteps=v_model_t / audio_latents=a_x_t / audio_timesteps=a_model_t / ut=v_u_t / aut=a_u_t.
_AWM_FLOW_TENSOR_KEYS = (
    "latents", "timesteps", "audio_latents", "audio_timesteps", "ut", "aut",
)

# Full leo.forward input contract replayed by the training recompute (forward_step): conditioning + flow.
_AWM_LEO_FORWARD_KEYS = _AWM_COND_TENSOR_KEYS + _AWM_FLOW_TENSOR_KEYS + _AWM_COND_LIST_KEYS



def get_post_train_video_dataloader(args, logger, text_encoder, text_encoder_2, dp_degree, dp_rank, local_seed=None):
    if args.post_train_type == "grpo" or args.post_train_type == "awm" or args.post_train_type == "refl":
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



class Leo2AWMFSDPTrainer(Leo2Trainer):
    """FSDP (pure torch) AWM trainer for Leo."""

    def __init__(self, args):

        # Debug fast-init: skip checkpoint loading
        # fewer steps for rollout
        if args._debug_fast_init:
            self._debug_fast_init_patch_args(args)
            args.sample_interval = -1
            args.rollout_sampling_steps = 3
            args.kl_weight = 0.0

        super().__init__(args)

        if not getattr(args, "output_dir", None):
            args.output_dir = str(Path(args.save).parent)

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

        #overwrite generation config with task specific kwargs if specified
        for key, value in args.t2vi2v_task_kwargs.items():
            if hasattr(self.model.generation_config, key):
                setattr(self.model.generation_config, key, value)


    def _debug_fast_init_patch_args(self, args):
        """Clear checkpoint-related args so `collect_load_plans` produces empty plan
        lists for the main DiT model (and, transitively, the ref model — it reads the
        same args). This skips HF/bin/dcp pretrained loads as well as resume-from-iter.
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
        """Skip main model dcp/resume checkpoint loads in debug fast-init mode.

        Defensive: with the args patched above, `after_fsdp_plans` should already be
        empty, so this guard is mainly to make the skip explicit in logs.
        """
        if getattr(self, "_debug_fast_init", False):
            self.logger.warning(
                "[DEBUG] LEO_DEBUG_FAST_INIT=1, skipping main model load_after_fsdp "
                "(dcp pretrained / resume from iter_*)"
            )
            return
        super().load_after_fsdp()

    def set_proxy(self):
        os.environ["http_proxy"] = "http://star-proxy.oa.com:3128"
        os.environ["https_proxy"] = "http://star-proxy.oa.com:3128"

    def set_api_key(self):
        os.environ["WANDB_API_KEY"] = "ffddb91f64606cb17216362faa7bc29540061a69"

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
        # Use unified reward interface (similar to flow_grpo)
        reward_model = getattr(args, "reward_model", "auto")
        if reward_model == "auto":
            reward_model = "videoalign_local"
        args.reward_model = reward_model
        
        # Create reward function using unified interface
        reward_inferencer = get_reward_fn(args, self.device, self.logger)
        self.logger.info(f"Using unified reward interface with model: {reward_model}")
        log_gpu_memory_usage(f"[memory] after loading reward model {reward_model}", logger=self.logger)
        return reward_inferencer

    def build_extra_model(self):
        
        self.logger.info('--> load extra models')
        args = self.args

        # ss maybe loaded from checkpoint, so only build when not exist.
        if not hasattr(self, "ss") or self.ss is None:
            self.ss = build_scalar_state()

        video_snr_type = args.flow_snr_type_video or args.flow_snr_type
        video_snr_mix_ratio = (
            args.flow_snr_mix_uniform_ratio_video
            if args.flow_snr_mix_uniform_ratio_video is not None
            else args.flow_snr_mix_uniform_ratio
        )

        self.video_denoiser = build_denoiser(
            denoiser_type="video",
            shift=args.flow_shift_video,
            snr_type=video_snr_type,
            snr_mix_uniform_ratio=video_snr_mix_ratio,
        )
        if args.audio_branch_model_name is not None:
            audio_snr_type = args.flow_snr_type_audio or args.flow_snr_type
            audio_snr_mix_ratio = (
                args.flow_snr_mix_uniform_ratio_audio
                if args.flow_snr_mix_uniform_ratio_audio is not None
                else args.flow_snr_mix_uniform_ratio
            )
            self.audio_denoiser = build_denoiser(
                denoiser_type="audio",
                shift=args.flow_shift_audio,
                snr_type=audio_snr_type,
                snr_mix_uniform_ratio=audio_snr_mix_ratio,
            )

        # vae, text
        self.vae = build_vae(only_encoder=False)
        if args.use_audio_vae:
            self.audio_vae = build_audio_vae()

        self.text_encoder = build_text_encoder()
        self.text_encoder_2 = None
        log_gpu_memory_usage("[memory] after loading vae and text encoder model", logger=self.logger)

        # reward model
        self.reward_inferencer = self.build_reward_model()

        # load ref model for kl
        self.load_ref_model()
        self.tokenizer = load_tokenizer(args.tokenizer_name, args.tokenizer_class)
        

    def after_initialize(self):
        pass

    # -- AWM resume hooks --------------------------------------------------
    def _collect_extra_client_state(self) -> dict:
        state = super()._collect_extra_client_state()
        # Number of data batches consumed in the current epoch, for exact
        # data-position resume (used together with the base-restored self.ss.epoch).
        # grpo_states themselves are persisted via scalar states (self.ss).
        state["awm_epoch_consumed_batches"] = getattr(self, "_epoch_consumed_batches", 0)
        return state

    def _consume_extra_client_state(self, client_state: dict) -> None:
        super()._consume_extra_client_state(client_state)
        # Stash consumed-batches count; applied at the top of the resumed epoch in
        # train() together with self.ss.epoch (restored by the base class).
        if client_state and "awm_epoch_consumed_batches" in client_state:
            self._resumed_epoch_consumed_batches = int(client_state.get("awm_epoch_consumed_batches", 0))
            self.logger.info(
                f"Staged AWM epoch_consumed_batches for resume: {self._resumed_epoch_consumed_batches}"
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
        # weights from `args.load`, never resume from a training checkpoint. When
        # `args.resume` is enabled, `collect_load_plans` would otherwise emit a
        # "resume" plan pointing at the latest `iter_*` policy checkpoint (and the
        # ref-model load loop below only handles "dcp" plans), leaving the reference
        # model without its correct weights. Temporarily disable resume for planning.
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
        return VideoPromptDataset(self.args, self.logger, self.args.t2vi2v_index_kwargs['trainsets'])

    @torch.no_grad()
    def sample_validation(self):
        args = self.args 
        run_task_kwargs = args.t2vi2v_task_kwargs
        index_task_kwargs = args.t2vi2v_index_kwargs

        # switch to evaluation mode
        self.model_engine.eval()

        # Overwrite generation config with task specific kwargs if specified.
        # IMPORTANT: t2vi2v_task_kwargs must include diff_guidance_scale. rollout_pipeline
        # mutates the shared generation_config.diff_guidance_scale to rollout_cfg_scale
        # (e.g. 1.0) and never restores it, so this loop is the only thing that resets it
        # back to the inference CFG before validation. Omitting it makes validation render
        # with no/wrong CFG from the 2nd validation interval onward.
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
    def prepare_samples_online(self, model, ref_model, reward_inferencer,
                                batch, global_step, dp_rank, sp_rank, sp_group, timesteps_train=None):
        """
        Prepare samples for online AWM training (Rollout Phase).
        
        This function performs the complete rollout process:
        1. Generate samples using the current policy model
        2. Compute rewards for generated samples
        3. Pre-compute reference model statistics (for KL regularization)
        
        The function supports two parallel group modes based on video_batch_size and num_generations:
        
        Mode 1: Single-rank groups (video_batch_size >= num_generations)
            - Each dp_rank processes complete group(s) locally
            - Example: video_batch_size=8, num_generations=4 -> 2 groups per rank
            - No cross-rank gather needed for rewards
            
        Mode 2: Multi-rank groups (video_batch_size < num_generations)
            - Multiple dp_ranks collaborate to complete one group
            - Example: video_batch_size=2, num_generations=8 -> 4 ranks per group
            - Cross-rank gather needed for reward computation
            
        CRITICAL Constraints:
            - If video_batch_size >= num_generations: video_batch_size % num_generations == 0
            - If video_batch_size < num_generations: num_generations % video_batch_size == 0
            Otherwise, samples within a rank may come from different prompts, breaking group structure!
        
        Args:
            model: Current policy model (for sample generation)
            ref_model: Reference model (for KL regularization, can be None)
            reward_inferencer: Reward model for computing rewards
            batch: Input batch containing prompts, seeds, etc.
            device: Device to run on
            global_step: Current training step
            dp_rank: Data parallel rank
            sp_rank: Sequence parallel rank (within SP group)
            sp_group: Sequence parallel groups
            
        Returns:
            videos: Generated videos tensor [Batch, C, F, H, W]
            audios: Decoded audios tensor [Batch, ...] or None (when audio was not decoded)
            reward_scores: Dictionary of reward scores for each sample
            all_latents: All latent states [Batch, Steps+1, C, H, W]
            all_log_probs: Log probabilities for each timestep [Batch, Steps]
            sigma_schedule: Noise schedule used for generation
            generation_info: Dictionary with generation metadata
            all_awm_sample_inputs: List of per-sample AWM recompute input dicts (mmdit kwargs aligned with LeoModel.forward; see _capture_awm_sample_inputs)
            all_ref_means: Pre-computed reference model means [Batch, Steps, ...] or None
        """
        args = self.args
        device = self.device
        world_size = self.world_size
        vae = self.vae
        # ========================================================================
        # Phase 0: Move Reward Model back to GPU (if previously offloaded)
        # ========================================================================
        # If reward models were offloaded to CPU in previous step, move them back to GPU
        # This is necessary because reward computation requires models to be on GPU
        reward_model_offload = getattr(args, "reward_model_offload", False)
        reward_model_load_time = 0.0
        if reward_model_offload:
            if hasattr(reward_inferencer, '_reward_models') and reward_inferencer._reward_models:
                rank = dist.get_rank() if dist.is_initialized() else 0
                load_start_time = sync_cuda_time()
                for reward_model in reward_inferencer._reward_models:
                    try:
                        # Handle two cases:
                        # 1. reward_model has a 'model' attribute (e.g., HPSv3RewardInferencer, VideoVLMRewardInference)
                        # 2. reward_model is the model itself (e.g., AltCLIPRM, CLIPModel)
                        model_to_move = None
                        if hasattr(reward_model, 'model') and reward_model.model is not None:
                            # Case 1: reward_model has a 'model' attribute
                            model_to_move = reward_model.model
                        elif hasattr(reward_model, 'parameters'):
                            # Case 2: reward_model is the model itself
                            model_to_move = reward_model
                        
                        if model_to_move is not None:
                            # Check if model is on CPU
                            params = list(model_to_move.parameters())
                            if params and not params[0].is_cuda:
                                # Move model back to GPU
                                model_to_move = model_to_move.to(device)
                                # Update the reference
                                if hasattr(reward_model, 'model'):
                                    reward_model.model = model_to_move
                                # Update device attribute if it exists
                                if hasattr(reward_model, 'device'):
                                    reward_model.device = device
                    except Exception as e:
                        self.logger.warning(f"Rank {rank}: Failed to move reward model back to GPU: {e}")
                load_end_time = sync_cuda_time()
                reward_model_load_time = load_end_time - load_start_time
                if reward_model_load_time > 0:
                    self.logger.info(f"Rank {rank}: Reward models loaded to GPU in {reward_model_load_time:.3f}s")
        
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
        scheduler.set_timesteps(num_inference_steps=args.rollout_sampling_steps, device=device)
        pipeline = self.create_pipeline(model, scheduler)
        sigma_schedule = scheduler.sigmas  # Store schedule for later use in training

        # ========================================================================
        # Phase 4: Initialize Storage Containers
        # ========================================================================
        all_videos = []       # Store generated videos: [Batch, C, F, H, W]
        all_audios = []       # Optional decoded audios: [Batch, ...] (empty when audio not decoded)
        reward_dicts = []     # Store reward dictionaries for each sample
        all_awm_sample_inputs = []  # Per-sample AWM recompute inputs (mmdit kwargs; see _capture_awm_sample_inputs)
        all_log_probs = []  # Store log probabilities: [Batch, Steps]



        # Set denoise level for latent reward
        if getattr(args, "reward_t", False):
            t_raw = args.reward_t * scheduler.num_train_timesteps
            timesteps_candidates = scheduler.timesteps.detach().cpu().tolist()
            # Snap to the nearest discrete timestep and return its index in the schedule
            sample_step = min(range(len(timesteps_candidates)), key=lambda i: abs(timesteps_candidates[i] - t_raw))
            use_latent_reward = True
            self.logger.info(f"Rank {rank} using latent reward at timestep {sample_step}")
        else:
            sample_step = None
            use_latent_reward = False

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
            batch_save_paths = []
            batch_group_info = []
            
            # For each sample in the mini-batch, prepare generation metadata
            for local_idx, sidx in enumerate(batch_indices):
                seed_value = seeds[sidx] if isinstance(seeds[sidx], (int, float)) else seeds[sidx].item()
                batch_seeds.append(seed_value)
                
                if ranks_per_group == 1:
                    group_idx_local = sidx // args.num_generations
                    sample_idx_in_group = sidx % args.num_generations
                    global_group_idx = dp_rank * num_groups_per_rank + group_idx_local
                else:
                    sample_idx_in_group = rank_in_group * video_batch_size + sidx
                    global_group_idx = group_idx
                
                save_dir = os.path.join(args.output_dir, "rl_samples", f"{global_step:07d}", f"group_{global_group_idx}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"index_{indexs[sidx]}_seed_{seed_value}_rank_{dp_rank}_sample_{sample_idx_in_group}.mp4")
                batch_save_paths.append(save_path)
                batch_generators.append(torch.Generator(device=device).manual_seed(seed_value))
                batch_group_info.append({
                    "sidx": sidx,
                    "group_idx_local": group_idx_local if ranks_per_group == 1 else None,
                    "global_group_idx": global_group_idx,
                    "sample_idx_in_group": sample_idx_in_group,
                })
            

            # --------------------------------------------------------------------
            # 5.2: Generate Videos with Current Policy Model
            # --------------------------------------------------------------------
            target_length = args.t2vi2v_task_kwargs["num_frames"]
            target_size = {256: (192, 336), 480: (352, 624), 640: (480, 848), 720: (544, 960), 960: (720, 1280), 1440: (1080, 1920)}
            target_height, target_width = target_size[args.video_bucket_hw_base_size]

            self.logger.info(f"Rank {rank} generating batch {batch_start_idx//mini_batch_size + 1}/{(num_prompts + mini_batch_size - 1)//mini_batch_size} "
                    f"(samples {batch_start_idx+1}-{batch_end_idx}/{num_prompts}), "
                    f"eta: {args.eta}, batch_size: {batch_size_actual}, "
                    f"{target_length}x{target_height}x{target_width}, flow_shift: {infer_flow_shift}, infer_steps: {args.rollout_sampling_steps}")
            
            # Configure determistic sampling for progressive training
            # For progressive training: use SDE (deterministic=False) for trainable timesteps, ODE (deterministic=True) for others
            # For "all" strategy (no mixgrpo): use SDE (deterministic=False) for all timesteps
            determistic = None
            single_sde_timestep_idx = None  # Will store the single SDE timestep index if using single SDE timestep selection
            training_strategy = getattr(args, 'training_strategy', 'all')
            use_single_sde_timestep = getattr(args, 'use_single_sde_timestep', False)
            num_inference_steps = args.rollout_sampling_steps
            
            # =================================================================
            # [LongCat] Single SDE timestep selection (Fix the stochastic timestep in SDE sampling)
            # When enabled, randomly select ONE timestep from [0, T') for SDE, all others use ODE
            # This is independent of grpo_states and timesteps_train
            # =================================================================
            if use_single_sde_timestep:
                # Determine T' (range from which to sample the single SDE timestep)
                sde_timestep_range = getattr(args, 'sde_timestep_range', None)
                if sde_timestep_range is None:
                    # Default to timesteps_group_size if not specified
                    sde_timestep_range = getattr(args, 'timesteps_group_size', num_inference_steps)

                # Initialize all timesteps to ODE (determistic=True)
                determistic = [True] * num_inference_steps
                
                # Randomly select ONE timestep from range [0, sde_timestep_range) for SDE
                # Sync across SP ranks to ensure consistency
                single_sde_timestep_idx = sync_random_tensor(
                    generator_fn=lambda: torch.randint(0, sde_timestep_range, (1,), device=device),
                    shape=(1,),
                    dtype=torch.long,
                    device=device,
                    sp_group=sp_group,
                ).item()
                
                # Set only the selected timestep to SDE (determistic=False)
                if single_sde_timestep_idx < num_inference_steps:
                    determistic[single_sde_timestep_idx] = False
                
                self.logger.info(f"Rank {rank} using LongCat single SDE timestep selection: "
                        f"single_sde_timestep_idx={single_sde_timestep_idx}, sde_timestep_range={sde_timestep_range}")
            elif training_strategy in ["progressive", "random", "decay", "dynamic"] and timesteps_train is not None:
                # Original progressive training: use SDE for all trainable timesteps
                # Initialize determistic list: True for all timesteps (ODE by default)
                determistic = [True] * num_inference_steps
                # Set False for trainable timesteps (use SDE for diversity)
                for timestep_i in timesteps_train:
                    if timestep_i < num_inference_steps:
                        determistic[timestep_i] = False
                
                self.logger.info(f"Rank {rank} using progressive training: timesteps_train={timesteps_train}, "
                        f"num_deterministic={sum(determistic)}/{len(determistic)}")
            elif training_strategy == "all":
                # When not using mixgrpo, use SDE (deterministic=False) for all timesteps
                determistic = False
                self.logger.info(f"Rank {rank} using 'all' strategy (no mixgrpo): determistic=False for all timesteps")
            elif training_strategy == "ode":
                # Use ODE (deterministic=True) for all timesteps
                determistic = True
                self.logger.info(f"Rank {rank} using 'ode' strategy (deterministic=True for all timesteps)")
            
            with torch.no_grad():
                # Batch generation: pass list of prompts to pipeline for parallel processing
                # Handle generator: use list if batch_size > 1, single generator if batch_size == 1
                generator_arg = batch_generators if batch_size_actual > 1 else batch_generators[0]
                # rollout_pipeline returns the conditional-only batched leo-forward kwargs
                # (batched_mmdit_kwargs, [B, ...]) -- no per-sample [cfg, ...] capture + cond
                # re-batch round-trip needed.
                pipeline_output, batched_mmdit_kwargs = self.rollout_pipeline(
                    pipeline,
                    model,
                    batch_prompts=batch_prompts,
                    batch_message_lists=batch_message_lists,
                    batch_seeds=batch_seeds,
                    target_height=target_height,
                    target_width=target_width,
                    target_length=target_length,
                    generator_arg=generator_arg,
                    determistic=determistic
                )
                videos_batch = pipeline_output.visuals
                batch_audios_batch = pipeline_output.audios
                batch_latents_batch = pipeline_output.all_latents
                batch_latents_audio_batch = pipeline_output.all_latents_audio
                batch_log_probs_batch, model_input_kwargs, _ = self.compute_log_prob(
                    model,
                    visual_latents=batch_latents_batch[:, -1],                # [B, C, T, H, W]
                    clean_audio_latents=(batch_latents_audio_batch[:, -1] if batch_latents_audio_batch is not None else None),                  # [B, C, L] or None
                    **batched_mmdit_kwargs, # {"cond_text": [B, ...], "condi_text_mask": [B, ...]....} refers to _AWM_COND_TENSOR_KEYS
                )

            # --------------------------------------------------------------------
            # 5.5: Store Generated Results
            # --------------------------------------------------------------------
            # Store per-sample decoded media + scalar log-prob (these accumulate one row per sample).
            for local_idx in range(batch_size_actual):
                all_videos.append(videos_batch[local_idx:local_idx+1])              # [1, C, F, H, W]
                if batch_audios_batch is not None:
                    all_audios.append(batch_audios_batch[local_idx:local_idx+1])  # [1, ...]
                all_log_probs.append(batch_log_probs_batch[local_idx:local_idx+1])  # [1,]

            all_awm_sample_inputs.append(model_input_kwargs)   # batched [b, ...] leo-forward kwargs {"latents": [B, 2C+1, T, H, W], "audio_latents": [B, C, L], "ut": [B, C, T, H, W], "aut": ...}


            # --------------------------------------------------------------------
            # 5.6: Compute Rewards (Only Rank 0 of SP Group)
            # --------------------------------------------------------------------
            # Only SP rank 0 computes rewards to avoid redundant computation
            if sp_rank == 0:
                batch_video_paths = []
                batch_pth_paths = []
                batch_images_for_reward = []
                
                for local_idx, (sidx, save_path) in enumerate(zip(batch_indices, batch_save_paths)):
                    videos_single = videos_batch[local_idx:local_idx+1]

                    # for leo2, output_type is pt, return tensor shape is (B, T/F, C, H, W), T short for time
                    # Save as PNG when generating single frame (image), otherwise save as video
                    if target_length == 1:
                        save_path_png = save_path.replace('.mp4', '.png')
                       
                        image_tensor = videos_single.squeeze(1)
                        torchvision.utils.save_image(image_tensor, save_path_png)
                        save_path = save_path_png
                        self.logger.info(f"Rank {rank}: Saved image {local_idx+1}/{batch_size_actual} to {save_path_png}")
                        batch_images_for_reward.append(image_tensor.squeeze(0))  # (C, H, W) for reward
                        batch_video_paths.append(None)
                    else:
                        # videos_single: (1, T, C, H, W) for leo2
                        # for leo2, video_processor.postprocess is already called in rollout_pipeline, we can directly save the video without additional postprocessing
                        # videos_single: (1, T, C, H, W) -> (T, C, H, W) -> (T, H, W, C) in uint8 format for save_video_audio
                        videos_single = (videos_single.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
                        save_video_audio(videos_single, None, save_path, fps=args.video_fps) # (T, H, W, C) in uint8 format for save_video_audio
                        self.logger.info(f"Rank {rank}: Saved video {local_idx+1}/{batch_size_actual} to {save_path}")
                        batch_video_paths.append(os.path.abspath(save_path))
                        batch_images_for_reward.append(None)
                
                with torch.no_grad():
                    if target_length == 1:
                        # For images, stack tensors and compute rewards in batch
                        images_tensor = torch.stack(batch_images_for_reward, dim=0)  # (batch_size, C, H, W)
                        scores_dict, meta_dict = reward_inferencer(images_tensor, batch_prompts, [{}] * batch_size_actual)
                    else:
                        # For videos, pass list of file paths
                        # Note: metadata is optional, but we pass it for consistency with image case
                        if use_latent_reward:
                            scores_dict, meta_dict = reward_inferencer(batch_video_paths, batch_prompts, [{}] * batch_size_actual, video_paths_latent=batch_pth_paths, use_latent_reward=True)
                        else:
                            scores_dict, meta_dict = reward_inferencer(batch_video_paths, batch_prompts, [{}] * batch_size_actual)
                
                for local_idx, (sidx, prompt, seed_value, save_path, group_info) in enumerate(zip(batch_indices, batch_prompts, batch_seeds, batch_save_paths, batch_group_info)):
                    reward_entry = {}
                    for k, v in scores_dict.items():
                        value = v[local_idx] if isinstance(v, list) and local_idx < len(v) else v
                        reward_entry[k] = value
                    
                    reward_dict_entry = {
                        "prompt": prompt,
                        "seed": seed_value,
                        "video_path": save_path,
                        "sample_idx": sidx,
                        "ranks_per_group": ranks_per_group,
                    }
                    
                    if ranks_per_group == 1:
                        reward_dict_entry.update({"group_idx_local": group_info["group_idx_local"], "global_group_idx": group_info["global_group_idx"], "sample_idx_in_group": group_info["sample_idx_in_group"]})
                    else:
                        reward_dict_entry.update({"group_idx": group_idx, "rank_in_group": rank_in_group, "sample_idx_in_group": group_info["sample_idx_in_group"]})
                    
                    reward_dict_entry.update(reward_entry)
                    reward_dicts.append(reward_dict_entry)
        
        # ========================================================================
        # Phase 6: Gather Rewards Across Ranks and Concatenate All Results
        # ========================================================================
        # Gather rewards from all ranks and process them for training
        reward_scores = gather_and_process_rewards(
            reward_dicts=reward_dicts,
            ranks_per_group=ranks_per_group,
            num_groups_per_rank=num_groups_per_rank,
            group_idx=group_idx,
            rank_in_group=rank_in_group,
            dp_rank=dp_rank,
            sp_rank=sp_rank,
            video_batch_size=video_batch_size,
            args=args,
            device=device,
            global_step=global_step,
            indexs=indexs,
            sp_group=sp_group,
            world_size=world_size,
            logger=self.logger,
        )
        self.logger.debug(f"Rank {rank}: Gathered rewards: {reward_scores}")
        # Expand reward scores to match timestep dimension for training
        reward_scores["ori_avg"] = reward_scores["avg"]  # Store original for logging
        reward_scores["avg"] = reward_scores["avg"].unsqueeze(1).repeat(1, args.rollout_sampling_steps)  # [Batch, Steps]
        
        # Concatenate all collected results into tensors / one batched dict.
        all_log_probs = torch.cat(all_log_probs, dim=0)  # [Total_Batch, Steps]
        all_awm_sample_inputs = self._concat_sample_inputs(all_awm_sample_inputs)  # batched dict [Total_Batch, ...] NOTE 这是一个dict
        videos = torch.cat(all_videos, dim=0)            # [Total_Batch, C, T, H, W]
        audios = torch.cat(all_audios, dim=0) if len(all_audios) > 0 else None


        # Clean up pipeline to free memory before reference model computation
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        # ========================================================================
        # Phase 6.5: Offload Reward Model to CPU (if enabled)
        # ========================================================================
        # Offload reward model to CPU after reward computation to save GPU memory
        # This helps prevent OOM during grpo_one_step training phase
        reward_model_offload = getattr(args, "reward_model_offload", False)
        reward_model_offload_time = 0.0
        if reward_model_offload:
            if hasattr(reward_inferencer, '_reward_models') and reward_inferencer._reward_models:
                self.logger.info(f"Rank {rank}: Offloading reward models to CPU...")
                offload_start_time = sync_cuda_time()
                for reward_model in reward_inferencer._reward_models:
                    try:
                        # Handle two cases:
                        # 1. reward_model has a 'model' attribute (e.g., HPSv3RewardInferencer, VideoVLMRewardInference)
                        # 2. reward_model is the model itself (e.g., AltCLIPRM, CLIPModel)
                        model_to_offload = None
                        if hasattr(reward_model, 'model') and reward_model.model is not None:
                            # Case 1: reward_model has a 'model' attribute
                            model_to_offload = reward_model.model
                        elif hasattr(reward_model, 'parameters'):
                            # Case 2: reward_model is the model itself
                            model_to_offload = reward_model
                        
                        if model_to_offload is not None:
                            # Check if model is on GPU before offloading
                            params = list(model_to_offload.parameters())
                            if params and params[0].is_cuda:
                                # Move model to CPU
                                model_to_offload = model_to_offload.to('cpu')
                                # Update the reference
                                if hasattr(reward_model, 'model'):
                                    reward_model.model = model_to_offload
                                # Update device attribute if it exists
                                if hasattr(reward_model, 'device'):
                                    reward_model.device = 'cpu'
                    except Exception as e:
                        self.logger.warning(f"Rank {rank}: Failed to offload reward model: {e}")
                torch.cuda.empty_cache()
                offload_end_time = sync_cuda_time()
                reward_model_offload_time = offload_end_time - offload_start_time
                self.logger.info(f"Rank {rank}: Reward models offloaded to CPU in {reward_model_offload_time:.3f}s")

        # Get mini-batch size for reference model computation (should match generation batch size)
        mini_batch_size = getattr(args, 'mini_batch_size_per_rollout', 1)
        if mini_batch_size <= 0:
            mini_batch_size = 1

        # ========================================================================
        # Phase 7: Pre-compute Reference Model Statistics (for KL Regularization)
        # ========================================================================
        # Pre-compute reference model statistics to avoid OOM during training
        # This computes mean predictions for all timesteps, stored on CPU to save GPU memory
        # Can be disabled by setting kl_compute_mode="training_phase" to compute on-the-fly during training
        all_ref_means = None
        ref_model_time = 0.0  # Track reference model computation time (CUDA-synced)
        
        # Get KL computation mode: "rollout_phase" (default, pre-compute) or "training_phase" (on-the-fly)
        kl_compute_mode = getattr(args, "kl_compute_mode", "rollout_phase")
        
        # Only pre-compute if:
        # 1. KL regularization is enabled
        # 2. Reference model is available
        # 3. kl_compute_mode is "rollout_phase" (default)
        if getattr(args, "kl_weight", 0.0) > 0 and ref_model is not None and kl_compute_mode == "rollout_phase":
            pass


        # ========================================================================
        # Phase 8: Finalize and Return Results
        # ========================================================================
        # Prepare generation metadata for training loop
        generation_info = {
            "generation_mode": generation_mode,
            "video_batch_size": video_batch_size,
            "ranks_per_group": ranks_per_group,
            "num_groups_per_rank": num_groups_per_rank,
            "group_idx": group_idx,
            "rank_in_group": rank_in_group,
            "samples_per_group": args.num_generations,
            "single_sde_timestep_idx": single_sde_timestep_idx,  # LongCat single SDE timestep (None if not using)
        }

        # Final memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Generate visualization HTML (only on rank 0 and sp_rank 0)
        if rank == 0 and sp_rank == 0:
            try:
                from pathlib import Path
                step_dir = Path(args.output_dir) / "rl_samples" / f"{global_step:07d}"
                if step_dir.exists():
                    generate_visualization_html(step_dir, global_step, self.logger, upload_to_cos=True)
            except Exception as e:
                self.logger.warning(f"Failed to generate visualization HTML: {e}")
        
        return videos, audios, reward_scores, all_awm_sample_inputs, all_log_probs, sigma_schedule, generation_info, ref_model_time, determistic, reward_model_load_time, reward_model_offload_time
    

    def _resolve_latent_channel_extend_type(self):
        """Read ``latent_channel_extend_type`` from the task kwargs, mirroring data_provider_dit.

        Only meaningful when ``extend_latent_channels`` is on; the value drives the channel-concat
        extension inside ``add_noise_and_extend_channel`` (t2v -> zero cond half + zero mask, i2v ->
        first frame, fl2v -> first+last frame). Must be one of ``SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES``.
        """
        task_kwargs = getattr(self.args, "t2vi2v_task_kwargs", {}) or {}
        latent_channel_extend_type = task_kwargs.get("latent_channel_extend_type")
        assert latent_channel_extend_type in SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES, (
            f"`latent_channel_extend_type` must be set in `t2vi2v_task_kwargs` to one of "
            f"{list(SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES)} when `extend_latent_channels=True`, "
            f"got {latent_channel_extend_type!r}."
        )
        return latent_channel_extend_type

    @staticmethod
    def _slice_cond_sample_inputs(model_inputs, n):
        """Slice the conditional (no-CFG) rows of ``model_inputs`` into one batched ``[n, ...]`` dict.

        prepare_model_inputs lays the batch out cfg-major as
        ``[cond_0..cond_{n-1}, uncond_0..uncond_{n-1}]`` (the unconditional half is present only
        when CFG is on), so the first ``n`` rows are exactly the conditional branch. The
        diffusion-loss recompute is a plain training forward with NO classifier-free guidance, so we
        keep those rows and drop the unconditional half -- in a single slice, with no per-sample
        capture + re-batch round-trip.

        Only the CONDITIONAL contract (``_AWM_COND_TENSOR_KEYS`` / ``_AWM_COND_LIST_KEYS``) is kept; the flow
        tensors (latents / ut / ...) are built fresh by compute_log_prob, NOT sliced here. A key absent /
        non-tensor in ``model_inputs`` is forwarded as ``None``. ``rope_media_info`` is a ``list`` (one entry
        per row), sliced as a list.
        """
        batched = {}
        for key in _AWM_COND_TENSOR_KEYS:
            v = model_inputs.get(key, None)
            batched[key] = v[:n].detach().clone() if isinstance(v, torch.Tensor) else None
        for key in _AWM_COND_LIST_KEYS:
            v = model_inputs.get(key, None)
            batched[key] = list(v[:n]) if v is not None else None
        return batched


    @staticmethod
    def _concat_sample_inputs(batched_dicts):
        """Concatenate per-mini-batch batched leo-forward kwargs dicts into ONE batched dict.

        Each element is the ``[b, ...]`` ``model_input_kwargs`` returned by ``compute_log_prob`` for
        one rollout mini-batch. They are joined along the sample dim so the result is a single
        ``[Total_Batch, ...]`` batched dict (directly usable as ``samples`` -- no per-sample list):
        tensors -> ``torch.cat(dim=0)``, ``rope_media_info`` (``list[b]``) -> list concat, scalars /
        ``None`` copied from the first chunk.
        """
        if not batched_dicts:
            return {}
        if len(batched_dicts) == 1:
            return batched_dicts[0]
        out = {}
        for k in batched_dicts[0].keys():
            vals = [d[k] for d in batched_dicts]
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                out[k] = torch.cat(vals, dim=0)
            elif isinstance(v0, list):
                merged = []
                for v in vals:
                    merged.extend(v)
                out[k] = merged
            else:
                out[k] = v0
        return out

    def compute_log_prob(
                        self, 
                        model, 
                        visual_latents=None, 
                        clean_audio_latents=None, *,
                        training=False, 
                        compute_ref=False,
                        **mmdit_kwargs
                        ):

        """Flow-matching log-prob surrogate (per-sample velocity MSE). Two modes:

        * ROLLOUT (``visual_latents`` / ``clean_audio_latents`` given): sample noise, build the leo.py
          forward inputs (mirrors the "prepare diffusion" block of ``data_provider_dit.prepare_model_inputs``,
          except the clean latents come from the rollout -- already normalized / model space), run the model,
          and return the reference log-prob + the frozen ``model_input_kwargs`` to store.
        * RECOMPUTE (both latents None): the frozen rollout forward inputs (latents / timesteps /
          audio_latents / audio_timesteps / ut / aut + mmdit kwargs) are passed back in via ``mmdit_kwargs``;
          we replay the exact same forward with the (updated) policy weights (NO re-noising) so the PPO ratio
          isolates the weight change.

        NOTE the ROLLOUT-only ``clean_audio_latents`` param is deliberately NOT named ``audio_latents``: the
        stored/replayed dict uses the ``audio_latents`` key (the *noised* a_x_t) which must fall through into
        ``mmdit_kwargs`` on the recompute path rather than binding this param.

        Args:
            model: the policy transformer. Run in ``train()`` mode so leo.py takes its loss branch.
            visual_latents: ROLLOUT only -- clean visual latents x1, ``[B, C, T, H, W]`` (conditional only).
            clean_audio_latents: ROLLOUT only -- clean audio latents x1, ``[B, C, L]``, or ``None``.
            **mmdit_kwargs: rollout -> conditional leo forward inputs; recompute -> the full frozen forward dict.

        Returns:
            (log_prob, model_input_kwargs): ``log_prob`` is the per-sample ``[B]`` velocity-MSE surrogate;
            ``model_input_kwargs`` is the leo.py forward dict actually fed to the model.
        """
        args = self.args
        device = model.device
        model.train()  # leo.py only enters its loss branch in train() mode

        # mmdit kwargs were captured during rollout and may live on CPU; move tensors to the compute
        # device (rope_media_info is a list[tuple], leave it as-is).
        mmdit_kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in mmdit_kwargs.items()
        }

        video_denoiser = get_video_denoiser()

        if visual_latents is not None or clean_audio_latents is not None:
            # ============================ ROLLOUT: noise fresh, build inputs ============================
            audio_denoiser = get_audio_denoiser() if clean_audio_latents is not None else None
            # ---- visual flow-matching forward + channel-concat extension, reusing the data-provider path ----
            latent_channel_extend_type = (
                self._resolve_latent_channel_extend_type() if getattr(args, "extend_latent_channels", False) else None
            )
            visual_latents = denormalize_vae_latents(self.vae, visual_latents.to(device))
            v_out = add_noise_to_latents(
                self.vae, visual_latents, device, denoiser=video_denoiser,
                sample_type="sample", noise_dtype=args.noise_dtype,
                latent_channel_extend_type=latent_channel_extend_type,
            )
            # x_0 (noise) / u_t (velocity target) stay at the gen-channel count; x_t carries the extended
            # channels for the model input. training_losses_fn uses x_0 / u_t (VELOCITY branch ignores x_t).
            v_t, v_model_t, v_noise, v_x_t, v_u_t = v_out.t, v_out.model_t, v_out.x_0, v_out.x_t, v_out.u_t

            # ---- audio flow-matching forward process (coupled to v_t unless --decouple-va-timestep) ----
            # Reuse the data-provider audio path (add_audio_noise). Unlike the video path it does NOT
            # re-normalize, which matches our model-space rollout audio latents, so feed them directly (no
            # denormalize). Audio has no channel extension and no flux shift, so n_tokens is irrelevant.
            a_model_t = a_x_t = a_u_t = None
            if clean_audio_latents is not None:
                audio_ts = None if getattr(args, "decouple_va_timestep", False) else v_t
                a_out = add_audio_noise(
                    self.audio_vae, clean_audio_latents, audio_denoiser,
                    sample_type="sample", noise_dtype=args.noise_dtype,
                    timesteps=audio_ts, device=device,
                )
                a_model_t, a_x_t, a_u_t = a_out.model_t, a_out.x_t, a_out.u_t

            # ---- assemble leo.py forward kwargs, mirroring data_provider's model_input_kwargs ----
            model_input_kwargs = dict(
                latents=v_x_t,                  # (extended) visual input  [B, 2c+1, t, h, w] or [B, c, t, h, w]
                timesteps=v_model_t,            # [B]
                audio_latents=a_x_t,            # noised audio input   [B, c, l] (or None)
                audio_timesteps=a_model_t,      # [B] (or None)
                ut=v_u_t,                       # velocity targets (leo.py / pp consume these)
                aut=a_u_t,
                **mmdit_kwargs,                 # rollout-captured leo forward inputs (conditional-only)
            )
            model_input_kwargs = {k: v for k, v in model_input_kwargs.items() if v is not None}

        else:
            # ===================== RECOMPUTE: replay the frozen rollout forward =====================
            # forward_step splats the stored dict back in via mmdit_kwargs (latents / timesteps /
            # audio_latents / audio_timesteps / ut / aut + conditional kwargs). No re-noising.
            model_input_kwargs = {k: v for k, v in mmdit_kwargs.items() if v is not None}
            v_u_t = model_input_kwargs.get("ut")
            a_u_t = model_input_kwargs.get("aut")
            a_x_t = model_input_kwargs.get("audio_latents")
            audio_denoiser = get_audio_denoiser() if a_x_t is not None else None

        # ---- forward: leo.forward only enters its loss branch in train() mode, where diffusion_loss_fn
        def _noop_loss_fn(model_output, **_):
            ref = model_output if isinstance(model_output, torch.Tensor) else model_output[0][0]
            return {"loss": ref.new_zeros(1)}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            output = model(
                **model_input_kwargs,
                diffusion_loss_fn=_noop_loss_fn,
                audio_diffusion_loss_fn=_noop_loss_fn,
            )

            if compute_ref: 
                with torch.no_grad():
                    ref_output = self.ref_model(
                        **model_input_kwargs, 
                        diffusion_loss_fn=_noop_loss_fn, 
                        audio_diffusion_loss_fn=_noop_loss_fn
                        )

                ref_v = dict(
                    ref_diffusion_prediction=ref_output.diffusion_prediction.detach(),
                    ref_audio_diffusion_prediction=(
                        ref_output.audio_diffusion_prediction.detach()
                        if ref_output.audio_diffusion_prediction is not None else None
                    ),
                    diffusion_prediction=output.diffusion_prediction,
                    audio_diffusion_prediction=output.audio_diffusion_prediction)
            else:
                ref_v = None
     
        # video log_prob is the negative of the video denoiser loss
        # FM loss = || v_theta(x_t, t) - (noise - samples) ||^2
        # log p_theta ≈ - FM loss + const
        # log N(y; mu_theta, sigma^2 I) = const - ||y - mu_theta||^2 / (2 sigma^2)
        log_prob = -video_denoiser.training_losses_fn(
            t=None, x0=None, xt=None, ut=v_u_t, model_output=output.diffusion_prediction,
        )["loss"]
        if getattr(args, "ratio_use_audio", False) and a_x_t is not None:
            audio_log_prob = -audio_denoiser.training_losses_fn(
                t=None, x0=None, xt=None, ut=a_u_t, model_output=output.audio_diffusion_prediction,
            )["loss"]
            log_prob = log_prob + audio_log_prob


        return log_prob, model_input_kwargs, ref_v


    def train_one_step(self, model, ref_model, reward_inferencer, sp_rank, sp_group, sp_size, dp_size,
                                batch, device, dp_rank, timesteps_train=None):
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
        
        # ==================== Phase 1: Online Sample Generation ====================
        rollout_start_time = sync_cuda_time()

        videos, audios, reward_scores, samples, all_log_probs, sigma_schedule, generation_info, ref_model_time, determistic, reward_model_load_time, reward_model_offload_time = self.prepare_samples_online(
            model, ref_model, reward_inferencer,
            batch, self.ss.update_steps, dp_rank, sp_rank, sp_group, timesteps_train=timesteps_train
        )
        rollout_end_time = sync_cuda_time()
        rollout_time = rollout_end_time - rollout_start_time
        
        # Log reward model offload/load times if enabled
        if getattr(args, "reward_model_offload", False):
            if reward_model_load_time > 0:
                logger.info(f"Rank {rank}: Reward model load time: {reward_model_load_time:.3f}s")
            if reward_model_offload_time > 0:
                logger.info(f"Rank {rank}: Reward model offload time: {reward_model_offload_time:.3f}s")
        
        # ==================== Phase 2: Prepare Training Data ====================
        batch_size = samples["latents"].shape[0]
        samples["log_probs"] = all_log_probs
    
        # ==================== Phase 3: Reward Processing and Advantage Computation ====================
        # Add reward scores to samples and gather statistics
        gathered_reward_stats = {}
        for metric_name, reward_tensor in reward_scores.items():
            reward_key = f"{metric_name}_rewards"
            samples[reward_key] = reward_tensor.to(torch.float32)
            gathered_reward_stats[metric_name] = gather_tensor(samples[reward_key])
            # Clean up NaN/inf values
            samples[reward_key] = torch.nan_to_num(samples[reward_key], nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract grouping info for advantage computation
        ranks_per_group = generation_info.get("ranks_per_group", 1)
        num_groups = generation_info.get("num_groups_per_rank", 1)
        samples_per_grp = generation_info.get("samples_per_group", args.num_generations)
        
        # Compute weighted advantages using Method 2: Separate Advantage First
        # Formula: A_total = w1 * Normalize(R1) + w2 * Normalize(R2)
        # This normalizes each reward separately (to unit variance), then weights them,
        # ensuring each reward contributes according to its weight regardless of original scale.
        reward_config = getattr(args, "reward_config", None)
        advantages = compute_weighted_advantages(
            samples=samples,
            reward_config=reward_config,
            ranks_per_group=ranks_per_group,
            num_groups=num_groups,
            samples_per_grp=samples_per_grp,
            logger=logger,
        )
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        # Expand advantages to match timestep dimension [batch, num_steps]
        # Individual rewards are 1D [batch], need to expand for per-timestep training
        if advantages.dim() == 1:
            num_timesteps = args.rollout_sampling_steps
            advantages = advantages.unsqueeze(1).repeat(1, num_timesteps)  # [batch, num_steps]
        samples["avg_advantages"] = advantages

        # For multi-rank groups, keep full-group rewards for normalization above,
        # then slice this rank's local chunk so shapes match local latents.
        if ranks_per_group > 1:
            chunk_start = generation_info.get("rank_in_group", 0) * batch_size
            chunk_end = chunk_start + batch_size
            reward_keys = [f"{metric_name}_rewards" for metric_name in reward_scores.keys()]
            for reward_key in reward_keys:
                samples[reward_key] = samples[reward_key][chunk_start:chunk_end]
            samples["avg_advantages"] = samples["avg_advantages"][chunk_start:chunk_end]

        # ==================== Phase 3.5: Successes and Filter Middle Advantage ====================
        # Initialize successes tensor (1 = valid sample, 0 = filtered sample)
        # This is used to filter out middle advantage samples, keeping only top and bottom
        successes = torch.ones(batch_size, dtype=torch.int32, device=device)
        
        # Filter middle percentage based on advantage values
        filter_middle_advantage = getattr(args, "filter_middle_advantage", False)
        filter_middle_ratio = getattr(args, "filter_middle_ratio", 0.5)  # Default: filter middle 50%
        num_generations = args.num_generations
        
        if filter_middle_advantage:
            # Gather all advantages from all DP ranks to get full group data
            # For advantages, we use the first timestep as representative (advantages are same across timesteps)
            # 
            # Key insight for SP: 
            # - In SP mode, multiple ranks (sp_size ranks) work on the SAME sample (different sequence parts)
            # - So we only need to gather across DP dimension, not across SP dimension
            # - dp_rank identifies which sample this rank is working on
            # - sp_rank identifies which part of the sequence this rank is processing
            #
            # Data layout: 
            # - world_size = dp_size * sp_size
            # - rank = dp_rank * sp_size + sp_rank
            # - All sp_ranks within same dp_rank have the same advantage (same sample)
            
            local_advantages = samples["avg_advantages"][:, 0].contiguous()  # [batch_size] - ensure contiguous
            
            # Only gather across DP dimension (sp_rank=0 ranks have the representative data)
            # In SP mode, all sp_ranks in the same dp_rank have the same advantage
            # We use global gather but then only use data from sp_rank=0 ranks
            gathered_advantages_all = gather_tensor(local_advantages).contiguous()  # [world_size * batch_size]
            
            # Extract only the advantages from sp_rank=0 ranks (one per dp_rank)
            # Layout after gather: [rank0_batch, rank1_batch, rank2_batch, ...]
            # Where rank = dp_rank * sp_size + sp_rank
            # We want: [dp_rank0_batch, dp_rank1_batch, ...] (only sp_rank=0)
            if sp_size > 1:
                # Reshape to [world_size, batch_size] then select sp_rank=0 rows
                gathered_advantages_all = gathered_advantages_all.view(world_size, batch_size)
                # Select only sp_rank=0 ranks: indices are [0, sp_size, 2*sp_size, ...]
                sp_rank_0_indices = torch.arange(0, world_size, sp_size, device=device)
                gathered_advantages = gathered_advantages_all[sp_rank_0_indices].contiguous().view(-1)  # [dp_size * batch_size]
            else:
                gathered_advantages = gathered_advantages_all.view(-1)  # [world_size * batch_size]
            
            # Reshape to [num_prompts, num_generations] for per-group processing
            total_samples = gathered_advantages.shape[0]  # dp_size * batch_size
            
            logger.info(f"[FilterMiddle] Rank {rank}: total_samples={total_samples}, num_generations={num_generations}, "
                    f"dp_size={dp_size}, batch_size={batch_size}, gathered_advantages={gathered_advantages.tolist()}")
            
            if total_samples % num_generations == 0:
                num_prompts = total_samples // num_generations
                gathered_advantages_reshaped = gathered_advantages.contiguous().view(num_prompts, num_generations)
                
                # Initialize gathered_successes (all 1s initially)
                gathered_successes = torch.ones(total_samples, dtype=torch.int32, device=device)
                gathered_successes_reshaped = gathered_successes.contiguous().view(num_prompts, num_generations)
                gathered_successes_modified = gathered_successes_reshaped.clone()

                
                for i in range(num_prompts):
                    group_advantages = gathered_advantages_reshaped[i]  # (num_generations,)
                    
                    # Sort all advantages (regardless of original success status)
                    sorted_indices = torch.argsort(group_advantages)
                    num_total = len(group_advantages)
                    
                    # Calculate boundaries: keep top and bottom, filter middle
                    keep_ratio = (1.0 - filter_middle_ratio) / 2.0  # e.g., 0.25 for 50% filter
                    top_start = int(num_total * (1.0 - keep_ratio))  # Top keep_ratio starts here
                    bottom_end = int(num_total * keep_ratio)  # Bottom keep_ratio ends here
                    
                    # Get indices of middle percentage (to be filtered)
                    middle_indices = sorted_indices[bottom_end:top_start]
                    
                    # Set success to 0 for middle percentage
                    if len(middle_indices) > 0:
                        gathered_successes_modified[i][middle_indices] = 0
                
                # Update gathered_successes (flatten back to 1D)
                gathered_successes = gathered_successes_modified.contiguous().view(-1)
                
                # Update local successes based on filtered gathered_successes
                # dp_rank identifies which slice of the gathered data belongs to this rank
                process_start = dp_rank * batch_size
                process_end = (dp_rank + 1) * batch_size
                successes = gathered_successes[process_start:process_end].clone()
                
                # Debug: verify the mapping is correct
                local_adv_for_verify = local_advantages.tolist()
                expected_adv_in_gathered = gathered_advantages[process_start:process_end].tolist()
                logger.info(f"[FilterMiddle] Rank {rank}: VERIFICATION - "
                        f"dp_rank={dp_rank}, process_range=[{process_start}:{process_end}], "
                        f"local_advantages={local_adv_for_verify}, "
                        f"expected_in_gathered={expected_adv_in_gathered}, "
                        f"match={local_adv_for_verify == expected_adv_in_gathered}, "
                        f"assigned_successes={successes.tolist()}")
                
            else:
                logger.warning(f"[FilterMiddle] Rank {rank}: total_samples ({total_samples}) not divisible by "
                            f"num_generations ({num_generations}), skipping filter_middle_advantage")
        
        # Store successes in samples for use in loss computation
        samples["successes"] = successes
        
        # ==================== Phase 4: Sample Shuffling ====================
        # Shuffle samples along batch dimension for better training stability
        # Note: This is generally beneficial even when using all samples, as it breaks potential
        # ordering biases and improves training stability. Cost is negligible.
        perm = sync_random_tensor(
            generator_fn=lambda: torch.randperm(batch_size, device=device),
            shape=(batch_size,),
            dtype=torch.long,
            device=device,
            sp_group=sp_group,
        )
        
        # Apply permutation to all samples
        reordered_samples = {}
        for k, v in samples.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                # Ensure perm is on the same device as v
                if v.device != perm.device:
                    perm_v = perm.to(v.device)
                else:
                    perm_v = perm
                reordered_samples[k] = v[perm_v]
            elif isinstance(v, list):
                perm_list = perm.cpu().tolist()
                reordered_samples[k] = [v[idx] for idx in perm_list]
            else:
                reordered_samples[k] = v
        samples = reordered_samples

        # ==================== Phase 5: Build Per-Sample / Per-MiniBatch Batches ====================
        # Support mini-batch processing along the sample dimension to improve throughput.
        # When mini_batch_size_per_update == 1, this reduces to the original per-sample behavior.
        mini_batch_size = getattr(args, "mini_batch_size_per_update", 1)
        if mini_batch_size <= 0:
            mini_batch_size = 1

        samples_batched_list = []
        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, batch_size)
            sample_dict = {}
            for k, v in samples.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    sample_dict[k] = v[start_idx:end_idx]                  # [MB, ...]
                elif isinstance(v, list):
                    sample_dict[k] = v[start_idx:end_idx]                  # list of MB entries
                else:
                    sample_dict[k] = v                                     # scalars / shared values
            samples_batched_list.append(sample_dict)
            
        # ==================== Phase 6: AWM Training Loop ====================
        kl_beta = getattr(args, "kl_weight", 0.0)
        clip_range = getattr(args, "clip_range", 1e-4)


        # Initialize metrics collection
        info = defaultdict(list)
        
        awm_loop_start_time = sync_cuda_time()
        update_successful, grad_norm = self.forward_backward(
            model,
            sp_rank,
            samples_batched_list,
            sigma_schedule,
            prompts,
            ref_image_paths,
            message_lists,
            info,
            mini_batch_size,
        )
        
        awm_loop_end_time = sync_cuda_time()
        awm_loop_time = awm_loop_end_time - awm_loop_start_time

        # ==================== Phase 8: Aggregate Metrics and Return ====================
        # Aggregate training metrics across all timesteps
        # Use appropriate aggregation for different metric types
        info_aggregated = {}
        
        # Auxiliary data for accurate computation of global std (parallel variance formula)
        # Need to collect: sum, sum_of_squares, count
        ratio_stats_for_reduce = {}  # {key: (sum, sum_sq, count)}
        
        for k, v in info.items():
            if k == "ratio_values":
                # Global ratio statistics: concatenate ratio values from all batches and timesteps
                all_ratios = torch.cat([r.flatten() for r in v])
                n = all_ratios.numel()
                
                # Directly computed statistics
                info_aggregated["ratio_max"] = all_ratios.max()
                info_aggregated["ratio_min"] = all_ratios.min()
                
                # Statistics for precise reduction: collect sum, sum_sq, count
                ratio_stats_for_reduce["ratio"] = (
                    all_ratios.sum(),
                    (all_ratios ** 2).sum(),
                    torch.tensor(float(n), device=all_ratios.device),
                    (all_ratios > 1.0).float().sum(),  # count of ratios > 1.0
                )
            elif k.startswith("ratio_values_t"):
                # Per-timestep ratio statistics
                timestep_idx = k.replace("ratio_values_t", "")
                all_ratios = torch.cat([r.flatten() for r in v])
                n = all_ratios.numel()
                
                # Compute per-timestep clip statistics
                clipped = (torch.abs(all_ratios - 1.0) > clip_range).float().sum()
                clipped_gt = (all_ratios - 1.0 > clip_range).float().sum()
                clipped_lt = (1.0 - all_ratios > clip_range).float().sum()
                
                ratio_stats_for_reduce[f"ratio_t{timestep_idx}"] = (
                    all_ratios.sum(),
                    (all_ratios ** 2).sum(),
                    torch.tensor(float(n), device=all_ratios.device),
                    clipped,
                    clipped_gt,
                    clipped_lt
                )
            else:
                # Other metrics (such as approx_kl, clipfrac, policy_loss, etc.)
                stacked = torch.stack(v)
                info_aggregated[k] = torch.mean(stacked)
        
        # Reduce accurate statistics (sum, sum_sq, count) and compute global mean and std
        for key, stats in ratio_stats_for_reduce.items():
            if key == "ratio":
                local_sum, local_sum_sq, local_count, local_gt1_count = stats
                
                # All-reduce sum, sum_sq, count, gt1_count
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_gt1_count, op=dist.ReduceOp.SUM)
                
                # Calculate global mean and std
                global_mean = local_sum / local_count
                global_var = (local_sum_sq / local_count) - (global_mean ** 2)
                global_std = torch.sqrt(torch.clamp(global_var, min=0.0))  # clamp to avoid negative due to numerical errors
                global_gt1_frac = local_gt1_count / local_count
                
                info_aggregated["ratio_mean"] = global_mean.item()
                info_aggregated["ratio_std"] = global_std.item()
                info_aggregated["ratio_gt_1_frac"] = global_gt1_frac.item()
            else:
                # Per-timestep ratio (ratio_t{idx})
                local_sum, local_sum_sq, local_count, local_clipped, local_clipped_gt, local_clipped_lt = stats
                
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_clipped, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_clipped_gt, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_clipped_lt, op=dist.ReduceOp.SUM)
                
                global_mean = local_sum / local_count
                global_var = (local_sum_sq / local_count) - (global_mean ** 2)
                global_std = torch.sqrt(torch.clamp(global_var, min=0.0))
                
                # Extract timestep index
                timestep_idx = key.replace("ratio_t", "")
                info_aggregated[f"ratio_mean_t{timestep_idx}"] = global_mean.item()
                info_aggregated[f"ratio_std_t{timestep_idx}"] = global_std.item()
                
                # Compute and record clipfrac per timestep
                info_aggregated[f"clipfrac_t{timestep_idx}"] = (local_clipped / local_count).item()
                info_aggregated[f"clipfrac_gt_one_t{timestep_idx}"] = (local_clipped_gt / local_count).item()
                info_aggregated[f"clipfrac_lt_one_t{timestep_idx}"] = (local_clipped_lt / local_count).item()
        
        # Reduce other metrics (use MAX or MIN for max/min, AVG for everything else)
        for key, value in info_aggregated.items():
            if isinstance(value, torch.Tensor):
                if key == "ratio_max":
                    dist.all_reduce(value, op=dist.ReduceOp.MAX)
                elif key == "ratio_min":
                    dist.all_reduce(value, op=dist.ReduceOp.MIN)
                else:
                    dist.all_reduce(value, op=dist.ReduceOp.AVG)
                info_aggregated[key] = value.item()
            else:
                info_aggregated[key] = value
        
        # Reduce timing metrics across ranks
        def reduce_time_metric(time_value):
            """Helper to reduce time metrics across ranks."""
            time_tensor = torch.tensor(time_value, device=device, dtype=torch.float32)
            dist.all_reduce(time_tensor, op=dist.ReduceOp.AVG)
            return time_tensor.item()
        
        rollout_time_avg = reduce_time_metric(rollout_time)
        awm_loop_time_avg = reduce_time_metric(awm_loop_time)
        ref_model_time_avg = reduce_time_metric(ref_model_time)
        
        dist.barrier()
        
        # Build return dictionary with all metrics
        # grad_norm is not equal between fsdp and megatron, fsdp grad_norm is 
        # the grad norm of 1/8 params, while megatron grad_norm is the grad norm
        # of all params across world_size dimensions
        return_dict = {
            "optimizer_step_successful": update_successful,
            "grad_norm": grad_norm,
            "KL_weight": kl_beta,
            "rollout_time": rollout_time_avg,
            "awm_loop_time": awm_loop_time_avg,
            "ref_model_time": ref_model_time_avg,
        }
        
        # Add aggregated training metrics
        return_dict.update(info_aggregated)
        
        # Add reward statistics (mean, std, max, min)
        for metric_name in reward_scores.keys():
            gathered = gathered_reward_stats.get(metric_name)
            if gathered is not None:
                return_dict[f"gathered_{metric_name}_reward_mean"] = torch.nanmean(gathered).item()
                return_dict[f"gathered_{metric_name}_reward_std"] = nanstd(gathered)
                # Add max and min for reward tracking
                valid_gathered = gathered[~torch.isnan(gathered)]
                if valid_gathered.numel() > 0:
                    return_dict[f"gathered_{metric_name}_reward_max"] = valid_gathered.max().item()
                    return_dict[f"gathered_{metric_name}_reward_min"] = valid_gathered.min().item()
        
        # Add advantages statistics (max, min)
        # advantages are stored in samples["avg_advantages"] with shape [batch, num_steps]
        all_advantages = samples["avg_advantages"]
        if all_advantages is not None and all_advantages.numel() > 0:
            return_dict["advantages_max"] = all_advantages.max().item()
            return_dict["advantages_min"] = all_advantages.min().item()
        
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
        sp_rank,
        samples_batched_list,
        sigma_schedule,
        prompts,
        ref_image_paths,
        message_lists,
        info,
        mini_batch_size,
    ) -> Tuple[bool, float]:
        args = self.args
        self.logger.info(f'sp_rank {sp_rank}: calling forward_backward, update step: {self.ss.update_steps}, train_steps: {self.ss.train_steps}')
        self.model_engine.optimizer.zero_grad()
        grad_norm = -1.0
        for mini_batch in self.mini_batch_iterator(samples_batched_list, mini_batch_size):
            with (
                profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)
                if getattr(args, 'profile', False) else contextlib.nullcontext()
            ) as prof:
                loss = self.forward_step(
                    sp_rank, sigma_schedule, prompts, ref_image_paths, message_lists,
                    info, mini_batch_size,
                    mini_batch, model,
                )
                # Scale loss by gradient accumulation steps
                final_loss = loss / args.gradient_accumulation_steps
                final_loss.backward()

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

    def forward_step(
        self, sp_rank, sigma_schedule: torch.Tensor, prompts, ref_image_paths, message_lists,
        info, mini_batch_size, mini_batch, model,
    ):  
        args = self.args
        
        # mini_batch is a dict from mini_batch_iterator. It carries the FROZEN rollout forward inputs
        # (_AWM_LEO_FORWARD_KEYS = conditional kwargs + flow inputs / velocity targets: latents / timesteps /
        # audio_latents / audio_timesteps / ut / aut). Splat them straight into compute_log_prob (visual_latents
        # / clean_audio_latents left None -> RECOMPUTE branch), which replays the exact same forward through the
        # policy model (NO re-noising). The remaining keys (log_probs / avg_advantages / successes / j /
        # *_rewards) are training-only and pulled out by name.
        recompute_kwargs = {
            k: mini_batch[k]
            for k in _AWM_LEO_FORWARD_KEYS
            if k in mini_batch
        }

        log_probs            = mini_batch["log_probs"].detach()
        avg_advantages       = mini_batch["avg_advantages"]
        successes            = mini_batch["successes"]
        j                    = mini_batch["j"] # default to -1
        kl_beta = getattr(args, "kl_weight", 0.0)
        clip_range = getattr(args, "clip_range", 1e-4)
        adv_clip_max = getattr(args, "adv_clip_max", 5.0)
        reference_model_offload = getattr(args, "reference_model_offload", False)

    
        # When audio is enabled, new_log_probs is the joint (visual + audio) log prob.
        if kl_beta > 0 and reference_model_offload:
            self.ref_model = self.ref_model.to(torch.cuda.current_device())

        kl_loss = None
        kl_loss_raw = None
        new_log_probs, _, ref_v = self.compute_log_prob(
            model, compute_ref=(kl_beta > 0), **recompute_kwargs
        )

        # ========== KL divergence computation (optional) ==========
        if kl_beta > 0:
            diff = ref_v["ref_diffusion_prediction"] - ref_v["diffusion_prediction"]
            kl_loss_raw = (diff ** 2).mean(dim=tuple(range(1, diff.ndim)), keepdim=False)
            if reference_model_offload:
                self.ref_model = self.ref_model.to('cpu')
                torch.cuda.empty_cache()

        # ========== Policy loss computation ==========
        # Clamp advantages to prevent extreme values
        advantages = torch.clamp(
            avg_advantages,
            -adv_clip_max,
            adv_clip_max,
        )
        
        # Compute probability ratio and PPO-style clipped loss
        ratio = torch.exp((new_log_probs - log_probs))

        with torch.no_grad():
            info["advantages"].append(advantages.detach().clone())
            info["ratio_values"].append(ratio.detach().clone())
            info[f"ratio_values_t{j}"].append(ratio.detach().clone())

        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss_per = torch.maximum(unclipped_loss, clipped_loss)

        # Apply successes mask to filter out middle advantage samples
        # successes: 1 = valid sample, 0 = filtered sample
        successes_mask = successes.float()  # [mini_batch_size]
        
        # Compute masked policy loss: only count valid samples
        # If all samples are filtered (sum=0), use mean to avoid division by zero
        num_valid_samples = successes_mask.sum()

        if num_valid_samples > 0:
            policy_loss = (policy_loss_per * successes_mask).sum() / num_valid_samples
        else:
            # Fallback: if no valid samples, use mean (this shouldn't happen normally)
            policy_loss = torch.mean(policy_loss_per)

        # Compute masked KL loss: apply same successes_mask for consistency
        # KL loss should only be computed on valid (non-filtered) samples
        if kl_beta > 0 and kl_loss_raw is not None:
            if num_valid_samples > 0:
                kl_loss = (kl_loss_raw * successes_mask).sum() / num_valid_samples
            else:
                kl_loss = torch.mean(kl_loss_raw)
        
        # Total loss: policy loss + KL regularization
        loss = policy_loss + kl_beta * kl_loss if (kl_beta > 0 and kl_loss is not None) else policy_loss
        
        # ========== Collect training metrics ==========
        log_prob_diff = new_log_probs.clone().detach() - log_probs.clone().detach()
        ratio_cloned = ratio.clone().detach()
        loss_cloned = loss.clone().detach()
        policy_loss_cloned = policy_loss.clone().detach()
        info["approx_kl"].append(0.5 * torch.mean(log_prob_diff ** 2))
        info["clipfrac"].append(torch.mean((torch.abs(ratio_cloned - 1.0) > clip_range).float()))
        info["clipfrac_gt_one"].append(torch.mean((ratio_cloned - 1.0 > clip_range).float()))
        info["clipfrac_lt_one"].append(torch.mean((1.0 - ratio_cloned > clip_range).float()))
        info["policy_loss"].append(policy_loss_cloned)
        info["total_loss"].append(loss_cloned)
        if kl_beta > 0 and kl_loss is not None:
            info["kl_loss"].append(kl_loss.clone().detach())
        return loss

    def mini_batch_iterator(self, samples_batched_list, mini_batch_size):
        """Yield per-step training mini-batches as dicts.

        Each yielded dict contains:
          * mmdit kwargs (cfg-major flattened to [cfg*MB, ...]) named after
            ``LeoModel.forward`` -- see ``_capture_awm_sample_inputs``.
            ``forward_step`` splats these straight into ``awm_denoise_step``.
          * Per-timestep slices: latents / next_latents / log_probs /
            prev_sample_mean / avg_advantages / timesteps / timestep_perms (+ optional
            ref_means, latents_audio / next_latents_audio / timesteps_audio).
          * ``successes`` mask and the inner timestep index ``j``.

        The cfg-major flatten ([B, cfg, ...] -> [cfg*B, ...] arranged as
        [cond_0..cond_{MB-1}, uncond_0..uncond_{MB-1}]) matches
        ``latent_model_input = cat([latents] * cfg)`` inside
        ``pipeline.denoise_step``.
        """
        inner_step = 0
        for _, sample in tqdm(
            list(enumerate(samples_batched_list)),
            desc=f"Global Step {self.ss.update_steps}: training",
            position=0,
            disable=self.rank >= 1,
        ):

            mb = dict(sample)
            # avg_advantages was broadcast to [MB, num_steps] (same value across steps); collapse to [MB].
            # timesteps is already the single frozen recompute step [MB] (leave as-is; do NOT slice).
            mb["avg_advantages"]   = sample["avg_advantages"][:, -1]
            mb["successes"]        = sample["successes"]
            mb["j"]                = -1
            yield mb
            inner_step += mini_batch_size

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

        # ============================== Build Video Dataset ==============================
        video_dataset = self.video_dataset
        video_sampler = self.video_sampler
        video_loader = self.video_loader

        # ============================== Initialize GRPO Training Strategy ==============================
        training_strategy = getattr(args, 'training_strategy', 'all')
        timesteps_group_size = getattr(args, 'timesteps_group_size', None)
        train_iters_per_timesteps_group = getattr(args, 'train_iters_per_timesteps_group', None)
        timesteps_group_overlap = getattr(args, 'timesteps_group_overlap', False)
        mixgrpo_stride = getattr(args, 'mixgrpo_stride', 1)
        
        # Calculate the number of training timesteps based on timestep_fraction
        num_timesteps = args.rollout_sampling_steps
        timestep_fraction = getattr(args, 'timestep_fraction', 1.0)
        num_train_timesteps = max(int(num_timesteps * timestep_fraction), 1)
        
        grpo_states = None
        use_single_sde_timestep_main = getattr(args, 'use_single_sde_timestep', False)
        
        # LongCat samples a random timestep t' from [0, T') each iteration, no window sliding needed
        if use_single_sde_timestep_main:
            sde_timestep_range = getattr(args, 'sde_timestep_range', None)
            if sde_timestep_range is None:
                sde_timestep_range = getattr(args, 'timesteps_group_size', num_train_timesteps)
            logger.info(f"--> [LongCat Single SDE Timestep] Skipping grpo_states, using fixed range [0, {sde_timestep_range}) for random t' sampling each iteration")
        elif training_strategy in ["progressive", "random", "decay", "dynamic"]:
            # Initialize GRPO training states for progressive training
            if timesteps_group_size is None or train_iters_per_timesteps_group is None:
                raise ValueError(f"timesteps_group_size and train_iters_per_timesteps_group must be set when using training_strategy={training_strategy}")
            
            grpo_states = GRPOTrainingStates(
                iters_per_group=train_iters_per_timesteps_group,
                group_size=timesteps_group_size,
                max_timesteps=num_train_timesteps,
                sample_strategy=training_strategy,
                overlap=timesteps_group_overlap,
                stride=mixgrpo_stride,
            )
            
            # Set additional parameters for decay/dynamic strategies
            if training_strategy == "decay":
                decay_kwargs = {
                    "max_iters_per_group": getattr(args, "max_iters_per_group", 100),
                    "min_iters_per_group": getattr(args, "min_iters_per_group", 25),
                }
                grpo_states.set_params(decay_kwargs)
            elif training_strategy == "dynamic":
                dynamic_kwargs = {
                    "dynamic_t1": getattr(args, "dynamic_t1", 12),
                    "dynamic_k": getattr(args, "dynamic_k", 0.5),
                    "dynamic_y0": getattr(args, "dynamic_y0", 5),
                }
                grpo_states.set_params(dynamic_kwargs)
            
            # Restore GRPO states from checkpoint if resuming
            if args.resume and args.resume != "None":
                grpo_states.restore_from_scalar_states(self.ss)
            
            logger.info(f"--> Initialized GRPO training strategy: {training_strategy}")
            logger.info(f"--> GRPO parameters: group_size={timesteps_group_size}, iters_per_group={train_iters_per_timesteps_group}, "
                    f"overlap={timesteps_group_overlap}, stride={mixgrpo_stride}, num_train_timesteps={num_train_timesteps}")
            logger.info(f"--> GRPO current state: cur_timestep={grpo_states.cur_timestep}, cur_iter_in_group={grpo_states.cur_iter_in_group}")
        
        
        args.gradient_accumulation_steps = args.gradient_accumulation_steps * args.video_micro_batch_size[-1]
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

            while True:
                #----------------- DATA LOADING -----------------#
                try:
                    batch = next(data_iter)
                    batch = sync_object_for_parallel_training(batch, parallel_dims=parallel_dims) # syncing is cheap: <0.01s
                except StopIteration:
                    break
                
                # for exact bacth-level resume
                self._epoch_consumed_batches += 1

                sync_random_states(parallel_dims)
                self.ss = sync_object_for_parallel_training(self.ss, parallel_dims=parallel_dims, force_object=True) # syncing is cheap: < 0.01s
                
                #----------------- SAMPLE VALIDATION -----------------#
                logger.info(f"Rank {dp_rank}: sample validation at update_steps {self.ss.update_steps}, sample_interval {args.sample_interval} {self.ss.update_steps % args.sample_interval == 0}")
                if args.sample_interval > 0 and (self.ss.update_steps > 0 and (self.ss.update_steps % args.sample_interval == 0)):
                    logger.info(f"Rank {dp_rank}: sample validation at update_steps {self.ss.update_steps}")
                    self.sample_validation()
                    # self.sample_validation(model, batch, self.ss.update_steps, dp_rank, sp_rank)
                    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

                #----------------- TRAINING -----------------#
                # Determine timesteps to train on for progressive training
                timesteps_train = None
                if grpo_states is not None:
                    timesteps_train = grpo_states.get_current_timesteps()
                    logger.info(f"Rank {dp_rank}: training on timesteps {timesteps_train} at update_steps {self.ss.update_steps}")

                step_start_time = sync_cuda_time()
                loss_dict = self.train_one_step(model, ref_model, self.reward_inferencer, sp_rank, sp_group, sp_size, dp_degree,
                                batch, device, dp_rank, timesteps_train=timesteps_train)
                step_end_time = sync_cuda_time()
                step_time = step_end_time - step_start_time

                # Update GRPO states after training step
                if grpo_states is not None:
                    grpo_states.update_iteration(seed=batch[2][0].item() if training_strategy == "random" else None)
                    # Update scalar states with GRPO states for checkpoint saving
                    grpo_states.update_scalar_states(self.ss)
                
                self.report_training_progress(step_time, loss_dict)

                #----------------- CHECKPOINTING -----------------#
                logger.info(f"--> ss.update_steps {self.ss.update_steps}, args.checkpointing_steps {args.checkpointing_steps}")
                if self.ss.update_steps % args.checkpointing_steps == 0 and self.ss.update_steps > 0:
                    logger.info(f"--> save checkpoint at step {self.ss.update_steps}, {args.output_dir}")
                    self.save_checkpoint()
                    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

        
            new_epoch = self.ss.inc_epoch()
            logger.info(f"Increase epoch to {new_epoch}.")

        self.save_checkpoint()


    def report_training_progress(self, sec_per_step, loss_dict):
        args = self.args
        #################### LOGGING ####################
        is_update_step = self.ss.train_steps % args.gradient_accumulation_steps == 0
        if is_update_step and self.ss.update_steps % args.log_interval == 0:
            # Simplified progress info - only log loss_dict contents
            progress_info = {
                "epoch": f"{self.ss.epoch}/{args.max_epochs}",
                "step": f"{self.ss.update_steps}",
                "learning_rate": f"{self.get_last_lr():.8f}",
                "step_time": f"{sec_per_step:.2f}s",
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
                self.write_events(summary_events)


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

    def build_audio_scheduler(self, device):
        """
        Build the audio AWM scheduler with the audio-specific flow-shift.

        ``set_timesteps`` is deterministic given (shift, num_inference_steps), so this reproduces exactly the
        audio noise schedule used by the rollout pipeline's audio scheduler, which lets the training-time
        log-prob recomputation stay consistent without threading the audio sigmas around.
        """
        scheduler = FlowMatchDiscreteScheduler(
            shift=self._audio_flow_shift(),
            reverse=True,
            solver="euler",
        )
        scheduler.set_timesteps(num_inference_steps=self.args.rollout_sampling_steps, device=device)
        scheduler._step_index = None
        scheduler._begin_index = None
        return scheduler

    def create_pipeline(self, model, scheduler):
        # Audio scheduler: built whenever an audio VAE is present, because for av / audio tasks the rollout
        # always runs the audio diffusion branch (independent of grpo_use_audio). Audio has its own flow-shift,
        # so its noise schedule differs from the visual one; the training-time log-prob recomputation uses a
        # matching audio schedule + audio timesteps (see build_audio_grpo_scheduler / samples["timesteps_audio"]).
        # The audio VAE / processor are passed so the rollout can decode audio by default.
        audio_scheduler = None
        audio_vae = model.model_dict.get("audio_vae") if hasattr(model, "model_dict") else None
        audio_processor = getattr(model, "audio_processor", None)
        if getattr(self.args, "use_audio_vae", False):
            audio_scheduler = FlowMatchDiscreteScheduler(
                shift=self._audio_flow_shift(),
                reverse=True,
                solver="euler",
            )
        return Leo2GRPOPipeline(
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
        num_frames = args.t2vi2v_task_kwargs.get("num_frames", args.num_frames)
        self.logger.info(f"prepare_model_inputs - message_list(len={len(batch_message_lists)})[0]: {batch_message_lists[0]}, media_size: {[target_height, target_width]}, num_frames: {num_frames}, {prepare_model_input_kwargs}")
        model_inputs = model.prepare_model_inputs(
            prompt=None, image=None, message_list=batch_message_lists, use_system_prompt=self.model.generation_config.use_system_prompt,
            seed=batch_seeds,
            num_frames=num_frames, **prepare_model_input_kwargs,
        )
        # The pipeline mutates model_inputs in-place (encode_prompt, pops input_ids /
        # channel_cond_vae_images, appends generation-loop state). Since
        # prepare_model_inputs is deterministic given the same message_list + seed,
        # keep a clean deepcopy now and reuse it for the training-time prompt
        # embeddings below instead of rebuilding it from scratch. torch.Generator is
        # not deepcopy-able and is not needed by the training path, so exclude it.
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
            "num_inference_steps": args.rollout_sampling_steps,
            "guidance_scale": guidance_scale,
            "generator": generator_arg,
            # For av, decode audio by default during rollout (the pipeline only decodes when an audio VAE is
            # present and the output_type carries an 'audio' entry).
            "output_type": dict(visual="pt", audio="pt") if is_av else "pt",
            "return_dict": True,
            "sde_type": args.sde_type,
            "eta": args.eta,
            "model_kwargs": model_inputs,
        }
        if is_av:
            pipeline_kwargs["audio_duration"] = model_inputs["batch_gen_audio_info"][0].audio_duration

        # if output_type is np, return np shape is (B, T/F, H, W, C)
        # if output_type is pt, return tensor shape is (B, T/F, C, H, W)
        # Add determistic parameter (always pass it, whether it's a list, False, or True)
        if determistic is not None:
            pipeline_kwargs["determistic"] = determistic

        pipeline_output = pipeline(**pipeline_kwargs)
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
        
        # NOTE: AWM Conditional-only reuse: the downstream diffusion-loss recompute is a plain training
        # forward with NO classifier-free guidance, so we only ever need the conditional rows.
        # prepare_model_inputs lays the batch out cfg-major as
        #   [cond_0..cond_{B-1}, uncond_0..uncond_{B-1}]
        # so the first B rows are exactly the conditional branch. Slice them once into a single
        # batched [B, ...] dict instead of capturing per-sample [cfg, ...] and re-batching later.
        B = len(batch_prompts)
        cond_mmdit_kwargs = self._slice_cond_sample_inputs(model_inputs, B)

        self.logger.info(
            f"cond_mmdit_kwargs: B={B}, "
            f"cond_text_states.shape={tuple(cond_mmdit_kwargs['cond_text_states'].shape)}, "
            f"cond_text_mask.shape={tuple(cond_mmdit_kwargs['cond_text_mask'].shape) if cond_mmdit_kwargs['cond_text_mask'] is not None else None}, "
            f"attention_mask.shape={tuple(cond_mmdit_kwargs['attention_mask'].shape) if cond_mmdit_kwargs['attention_mask'] is not None else None}, "
            f"rope_media_info.len={len(cond_mmdit_kwargs['rope_media_info']) if cond_mmdit_kwargs['rope_media_info'] is not None else 0}"
        )
        # Return the full pipeline output (callers access fields on demand, e.g. .visuals / .all_latents /
        # .all_log_probs / .all_prev_means and the optional audio-side .all_latents_audio / ...) plus the
        # conditional-only batched mmdit kwargs. Channel-cond images are rebuilt inside compute_log_prob
        # from latent_channel_extend_type, so they are no longer threaded through here.
        return pipeline_output, cond_mmdit_kwargs

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


def _compute_group_gate(
    semantic_rewards: torch.Tensor,
    ranks_per_group: int = 1,
    num_groups: int = 1,
    samples_per_grp: Optional[int] = None,
    quantile: float = 0.5,
    temp_scale: float = 1.0,
    abs_floor: Optional[float] = None,
) -> torch.Tensor:
    """Per-group soft gate derived from raw semantic rewards.

    For each GRPO group, the gate is::

        gate = sigmoid((R - tau_g) / (std_g * temp_scale))

    where ``tau_g`` is the per-group ``quantile`` of the (finite) semantic
    rewards and ``std_g`` is the per-group std (so the temperature self-scales
    with the reward magnitude). The optional ``abs_floor`` hard-zeros any sample
    whose semantic reward is below it. Non-finite entries map to gate 0.

    The grouping logic mirrors ``_normalize_group`` exactly so the gate aligns
    1:1 with the per-metric normalized advantages. Designed for 1D ``[batch]``
    reward tensors (the shape used in this trainer); 2D inputs are handled by
    pooling quantile/std over all finite entries of the group.

    Returns a tensor in ``[0, 1]`` with the same shape as ``semantic_rewards``.
    """
    gate = torch.zeros_like(semantic_rewards)
    temp_scale = max(float(temp_scale), 1e-6)

    def _gate_values(values: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(values)
        finite = torch.isfinite(values)
        if not finite.any():
            return result
        valid = values[finite]
        tau = torch.quantile(valid, quantile)
        temp = torch.clamp(valid.std(unbiased=False), min=1e-6) * temp_scale
        g = torch.sigmoid((values - tau) / temp)
        if abs_floor is not None:
            g = g * (values >= abs_floor).to(g.dtype)
        g = torch.where(finite, g, torch.zeros_like(g))
        return g

    if ranks_per_group == 1:
        if samples_per_grp is None:
            samples_per_grp = (
                len(semantic_rewards) // num_groups if num_groups > 0 else len(semantic_rewards)
            )
        for grp_idx in range(num_groups):
            start_idx = grp_idx * samples_per_grp
            end_idx = start_idx + samples_per_grp
            slice_obj = slice(start_idx, min(end_idx, len(semantic_rewards)))
            if slice_obj.start >= slice_obj.stop:
                continue
            gate[slice_obj] = _gate_values(semantic_rewards[slice_obj])
    else:
        # Multi-rank groups: rewards already gathered for the full group.
        gate = _gate_values(semantic_rewards)

    return gate


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

    # ---- Motion gating (optional) ---------------------------------------
    # When enabled, the advantage of "motion" metrics is multiplied by a
    # per-group soft gate derived from the "semantic" metric, so that motion
    # only gets rewarded for samples that are already semantically acceptable
    # ("semantic first: if semantics is wrong, motion is meaningless").
    mg_cfg = (reward_config.get("motion_gating") or {}) if isinstance(reward_config, dict) else {}
    mg_enable = bool(mg_cfg.get("enable", False))
    mg_semantic = str(mg_cfg.get("semantic_metric", "TA")).lower()
    mg_motion = {str(m).lower() for m in mg_cfg.get("motion_metrics", ["MQ"])}
    mg_quantile = float(mg_cfg.get("quantile", 0.5))
    mg_temp_scale = float(mg_cfg.get("temp_scale", 1.0))
    mg_abs_floor = mg_cfg.get("abs_floor", None)
    mg_abs_floor = None if mg_abs_floor is None else float(mg_abs_floor)

    motion_gate = None
    if mg_enable:
        # Locate the raw semantic reward tensor across all models by suffix.
        sem_key = next(
            (
                k for k in samples
                if k.endswith(f"_{mg_semantic}_rewards")
                and k not in ("avg_rewards", "ori_avg_rewards")
            ),
            None,
        )
        if sem_key is None:
            logger.warning(
                f"Motion gating enabled but no semantic reward key matching "
                f"'*_{mg_semantic}_rewards' found in samples; disabling gating."
            )
        else:
            motion_gate = _compute_group_gate(
                samples[sem_key],
                ranks_per_group=ranks_per_group,
                num_groups=num_groups,
                samples_per_grp=samples_per_grp,
                quantile=mg_quantile,
                temp_scale=mg_temp_scale,
                abs_floor=mg_abs_floor,
            )
            logger.info(
                f"Motion gating ON: semantic='{mg_semantic}' (key={sem_key}), "
                f"motion={sorted(mg_motion)}, quantile={mg_quantile}, "
                f"temp_scale={mg_temp_scale}, abs_floor={mg_abs_floor}, "
                f"gate.mean={motion_gate.mean().item():.4f}, "
                f"gate.min={motion_gate.min().item():.4f}, "
                f"gate.max={motion_gate.max().item():.4f}"
            )

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
                contribution = combined_weight * normalized_advantage
                # Gate motion metrics by the semantic gate (if enabled): the
                # motion advantage only counts where semantics is acceptable.
                if motion_gate is not None and metric_name.lower() in mg_motion:
                    contribution = contribution * motion_gate
                weighted_advantages = weighted_advantages + contribution
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

def build_grpo_scheduler(args, sigma_schedule, device):
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

def move_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, dict):
        return {k: move_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_to_cpu(v) for v in x]
    else:
        return x