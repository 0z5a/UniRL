import gc
import os
import random
import time
from collections import defaultdict, OrderedDict
from typing import Dict, Union, List, Optional

import loguru
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
 
from .helpers import (
    CycleStates,
    GRPOTrainingStates,
    save_checkpoint,
)
from .multimodal_gemini_beta_grpo_trainer import MultiModalGeminiBetaGRPOTrainer
from ..constants import C_SCALE
from ..data_kits.rl_t2i_loader import RLTextImageArrowStream
from ..data_kits.samplers import RepeatRandomDistributedSampler
from ..ds_config import get_deepspeed_config
from ..models import build_model
from ..samplers.gemini_beta_sampler import GeminiBetaSampler
from ..utils.torch_utils import (
    set_manual_seed,
    set_worker_seed_builder,
    PRECISION_TO_TYPE,
)
from ..utils.deepspeed_utils import unwrap_model_for_generation_deepspeed
from ..utils.rl_exploration_metric import adaptive_noise_scale
from ..utils.torch_distributions import gather_tensor
from ..utils.torch_utils import profiler_context
from hymm.data_kits.system_prompt import unified_system_prompt_en, t2i_system_prompts
from hymm.parallelism.parallel_states import init_parallel_state

gc.set_threshold(7000, 100, 100)


def compute_multimodal_grpo_loss(
        advantages: torch.Tensor,
        log_prob: torch.Tensor, 
        old_log_probs: torch.Tensor,
        adv_clip_max: float,
        clip_range: float,
        kl_weight: float,
        prev_sample_mean: torch.Tensor,
        prev_sample_mean_ref: torch.Tensor,
        std_dev_t: float,
        text_loss_mask: Optional[torch.Tensor]=None,
        text_logits: Optional[torch.Tensor]=None,
        text_ref_logits: Optional[torch.Tensor]=None,
        kl_weight_text: Optional[float]=None,
        loss_mode: str="image_only",  # "image_only", "text_only", "both"
        only_kl_loss: bool=False,
):
    """
    Compute the multimodal grpo loss with configurable loss modes.
    
    Args:
        advantages (torch.Tensor): The advantages tensor.
        log_prob (torch.Tensor): The log probability tensor of the image sample.
        old_log_probs (torch.Tensor): The old log probability tensor of the image sample.
        adv_clip_max (float): The clip max value for advantages.
        clip_range (float): The clip range value.
        kl_weight (float): The KL weight value.
        prev_sample_mean (torch.Tensor): The previous sample mean tensor of the image sample.
        prev_sample_mean_ref (torch.Tensor): The previous sample mean reference tensor of the image sample.
        std_dev_t (float): The standard deviation tensor of the image sample.
        text_loss_mask (Optional[torch.Tensor]): The text loss mask tensor.
        text_logits (Optional[torch.Tensor]): The text logits tensor.
        text_ref_logits (Optional[torch.Tensor]): The text reference logits tensor.
        kl_weight_text (float): The KL weight value for text.
        loss_mode (str): Loss computation mode:
            - "image_only": Only compute image policy loss (default diffusion loss)
            - "text_only": Only compute text policy loss
            - "both": Compute both image and text losses (default)
        only_kl_loss (bool): Whether to only compute KL loss.
    """
    torch.cuda.empty_cache()
    
    # Clip advantages
    advantages = torch.clamp(
        advantages,
        -adv_clip_max,
        adv_clip_max,
    )
    
    # Initialize losses
    policy_loss_image = torch.tensor(0.0, device=advantages.device)
    policy_loss_text = torch.tensor(0.0, device=advantages.device)
    kl_loss_image = torch.tensor(0.0, device=advantages.device)
    kl_loss_text = torch.tensor(0.0, device=advantages.device)
    
    # ===== Image Policy Loss =====
    if loss_mode in ["image_only", "both"]:
        ratio = torch.exp(log_prob - old_log_probs)  # FIXME: 需要修改
        unclipped_loss = -advantages.detach() * ratio
        clipped_loss = -advantages.detach() * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        policy_loss_image = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
    
    # ===== Text Policy Loss =====
    if loss_mode in ["text_only", "both"] and text_logits is not None:
        # text_logits: [1, sequence_length-1, vocab_size]
        # text_loss_mask: [1, sequence_length-1]
        
        # First filter the valid portion using a mask, then compute the loss.
        if text_loss_mask is not None:
            assert text_loss_mask.shape == (1, text_logits.shape[1]), \
            f"The shape of text_loss_mask should be (1, {text_logits.shape[1]}), " \
            f"but got {text_loss_mask.shape}."

            # Only compute loss on the valid positions using the mask, to reduce memory usage
            valid_mask = text_loss_mask.bool()
            if valid_mask.any():
                # Extract the valid text_log_probs and the corresponding advantages
                valid_text_logits = text_logits[valid_mask]
                valid_text_log_probs = valid_text_logits.log_softmax(dim=-1)

                # Only compute ratio and loss on the valid positions
                ratio_text_valid = torch.exp(valid_text_log_probs - valid_text_log_probs.detach())
                unclipped_loss_text_valid = -advantages.detach() * ratio_text_valid
                clipped_loss_text_valid = -advantages.detach() * torch.clamp(
                    ratio_text_valid,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )
                policy_loss_text = torch.mean(torch.maximum(unclipped_loss_text_valid, clipped_loss_text_valid))
            else:
                policy_loss_text = torch.tensor(0.0, device=text_logits.device)
        else:
            # If there is no mask, compute the loss in the original way
            valid_text_logits = text_logits
            ratio_text = torch.exp(valid_text_logits - valid_text_logits.detach())
            unclipped_loss_text = (-advantages.detach() * ratio_text)
            clipped_loss_text = -advantages.detach() * torch.clamp(
                ratio_text,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            policy_loss_text = torch.mean(torch.maximum(unclipped_loss_text, clipped_loss_text))
    
    # ===== Combine Policy Losses =====
    policy_loss = policy_loss_image + policy_loss_text
    
    # ===== KL Loss =====
    # Image KL loss
    if kl_weight > 0 and loss_mode in ["image_only", "both"]:
        kl_loss_image = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t ** 2)
        kl_loss_image = torch.mean(kl_loss_image)
    
    # Text KL loss (as in grpo report)
    if kl_weight_text is not None and kl_weight_text > 0 and loss_mode in ["text_only", "both"]:
        assert text_logits.shape == text_ref_logits.shape, \
            f"The shape of text_logits and text_ref_logits should be the same, " \
            f"but got {text_logits.shape} and {text_ref_logits.shape} respectively."

        if text_loss_mask is not None:
            valid_text_ref_logits = text_ref_logits[valid_mask].detach()
        else:
            valid_text_ref_logits = text_ref_logits.detach()
        kl_loss_text = torch.exp(valid_text_ref_logits - valid_text_logits) - (valid_text_ref_logits - valid_text_logits) - 1
        kl_loss_text = torch.mean(kl_loss_text)
    
    # Combine KL losses
    kl_weight_text_effective = kl_weight_text if kl_weight_text is not None else kl_weight
    kl_loss = kl_loss_image * kl_weight + kl_loss_text * kl_weight_text_effective
    
    # Final loss
    if kl_weight > 0 or (kl_weight_text is not None and kl_weight_text > 0):
        loss = policy_loss + kl_loss
        if only_kl_loss:
            loss = kl_loss * 0.1
    else:
        loss = policy_loss
    
    return loss, policy_loss, kl_loss


class MultiModalGeminiT2TIBetaGRPOPureTorchParallelTrainer(MultiModalGeminiBetaGRPOTrainer):
    def __init__(self, args, all_dataset_keys=None):
        self.dp_rank_is_correct = False # 继承太多层，没有仔细看各个函数的调用顺序，增加这个flag，确保在用到dp_rank的时候都是正确的
        super().__init__(args)

        self.args.t2i_system_prompt = {
            'en_unified': unified_system_prompt_en,
            'en_vanilla': t2i_system_prompts['en_vanilla'],
            'en_think_recaption': t2i_system_prompts['en_think_recaption'],
            'en_recaption': t2i_system_prompts['en_recaption'],
            'none': None,
        }[args.t2i_system_prompt_type]

        assert args.launcher == 'pure_torch'
    
    def build_sampler(self):
        # TODO: Configuration
        self.args.t2i_pred_text_mode = "cot_recaption"
        self.args.cot_max_length = 768

        model_dict = dict(
            ref_model=self.ref_model,
            model_settings=self.model_settings,
        )
        # model在推理时(prepare_model_inputs)通过unwarp self.model_engine定义即可
        factor_kwargs = {"device": self.device, "dtype": PRECISION_TO_TYPE[self.args.precision]}
        self.model_dict = GeminiBetaSampler.build_extra_model(
            self.args,
            model_dict,
            factor_kwargs,
            logger=self.logger,
        )
        self.sampler = GeminiBetaSampler(
            self.args,
            model_dict=self.model_dict,
            pipeline_name=self.pipeline_name,
            rank=self.dp_rank,
            world_size=self.dp_size,
            device=self.device,
            logger=self.logger,
        )

    def init_distributed_env(self):
        super().init_distributed_env()
        # todo: initialize two meshes
        self.parallel_dims = init_parallel_state(
            dp_replicate=-1, # world_size//8 在单个Node下进行切片
            # dp_replicate=dist.get_world_size()// 32, # 在单个Node下进行切片
            dp_shard=-1,
            sp=1,
            tp=1,
            pp=self.args.pp_size,
            ep=self.args.ep_size,
            world_size=dist.get_world_size(),
        )
        self.parallel_dims.build_mesh('cuda')
        self.dp_rank = self.parallel_dims.dp_mesh.get_local_rank()
        self.dp_size = self.parallel_dims.dp_mesh.size()
        self.dp_rank_is_correct = True
        # assert self.parallel_dims.pp_enabled, '不开pp还要用这个trainer要改造一下 train_step'

    def task_init(self, args, all_dataset_keys=None):
        super().task_init(args, all_dataset_keys)
        self.update_ref_model = args.get("update_ref_model", False)
        self.ema_update_ref_steps = args.get("ema_update_ref_steps", 50)
        self.ema_update_ref_decay = args.get("ema_update_ref_decay", 0.95)
        self.mixgrpo_cur_timesteps = args.get("mixgrpo_cur_timesteps", 0)

        # text grpo logic related
        self.kl_weight_text = args.get("kl_weight_text", 0.0)
        # TODO
        self.loss_mode = args.get("loss_mode", "image_only")
        if self.loss_mode in ["text_only", "both"] and self.kl_weight_text > 0:
            # 目前是在 pipeline 中reference model直接计算ref logits，所以不支持kv-cache，正常方式应该用 reference model
            assert not args.kv_cache, "kv-cache is not supported for computing text KL loss"
        self.gen_text_freeze_for_same_prompt = args.get("gen_text_freeze_for_same_prompt", True)
        
        # Adaptive sde noise scale related
        self.update_sde_noise_scale = args.get("update_sde_noise_scale", False)
        self.exploration_score = None
        self.baseline_exploration = None
        self.initial_noise_scale = args.get("sde_noise_scale", 0.7)
        self.min_noise_scale = args.get("sde_noise_scale", 0.7)
        self.max_noise_scale = self.min_noise_scale * 2.0
        self.sde_noise_scale = self.initial_noise_scale

    def initialize_puretorch_model_engine(self):
        from hymm.parallelism.engines.gemini_parallel import GeminiParallelEngine
        from torch.optim import Adam, AdamW
        if (self.parallel_dims.ep or self.parallel_dims.pp) and self.args.get('pretrained_ckpt', None):
            load_ckpt_path = self.args.get('pretrained_ckpt', None)
        else:
            load_ckpt_path = None

        ds_config = get_deepspeed_config(self.args)
        if self.args.tensorboard:
            ds_config["tensorboard"] = {
                "enabled": True,
                "output_path": str(self.output_dir.absolute()),
                "job_name": self.exp_dir.name,
            }

        def scalar_state_to_dict(scalar_state):
            scalar_state_dict = scalar_state.to_dict()
            # sync scalar_state
            if dist.is_available() and dist.is_initialized():
                gather_results_list = [None for _ in range(dist.get_world_size())]
                torch.distributed.all_gather_object(gather_results_list, scalar_state_dict)
                scalar_state_dict = gather_results_list

            client_state = {
                "config": self.args,
                "scalar_state": scalar_state_dict,
            }
            return client_state
        
        self.logger.info(f"!!!!!! Must check the gradient_accumulation_steps: {self.args.gradient_accumulation_steps}")
        self.model_engine = GeminiParallelEngine(
            model=self.model,
            ds_config=ds_config,
            load_ckpt_path=load_ckpt_path,
            # auto_benchmark=True,
            # training_benchmark_schedule=('1F1B',),
            # pipeline_parallel_schedule='1F1B',
            m_microbatch=2,
            # micro_batch_size=2, # grpo micro_batchsize should be 2
            optimizer_config=dict(
                optimizer_cls={'AdamW': AdamW, 'Adam': Adam}[self.args.optimizer_name],
                optimizer_kwargs=dict(
                    lr=self.args.optimizer_params['lr'],
                    betas=self.args.optimizer_params['betas'],
                    weight_decay=self.args.optimizer_params['weight_decay'],
                    eps=self.args.optimizer_params['eps'],
                )
            ),
            pp_enable_autocast=self.args.autocast_dtype != 'fp32',
            # pp_enable_autocast=False,
            autocast_prec=self.args.autocast_dtype,
            weight_prec=self.args.precision,
            # cpu_offload=True,
            initial_training_states=scalar_state_to_dict(self.get_states_cls('scalar')()),
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

        # Resume from checkpoint
        # Must guarantee the `args.resume_puretorch` is set. Otherwise, the loading process will raise error
        # that "Missing key in checkpoint state_dict: states.config.resume_puretorch"
        assert (
            hasattr(self.args, 'resume_puretorch')
        ), f"args.resume_puretorch is not set, please set it to 'False' if you don't want to resume from checkpoint"
        self.resume_puretorch = self.args.resume_puretorch
        if self.resume_puretorch:
            _, training_states = self.model_engine.load_checkpoint(self.resume_puretorch, load_optimizer_states=True)
            client_state = training_states['scalar_state']
            self.logger.info(f"scalar_state: {client_state[0]}")

            # Resume ScalarStates. Overlap the initial states.
            self.ss = self.get_states_cls('scalar').from_pretrained(
                client_state[0],  # directly use the first-rank's scalar_state, since all ranks have the same scalar_state
                rank=self.rank,
                world_size=self.world_size,
                default_rank0_ss=self.args.default_rank0_ss,
                default={'lr': self.args.lr},
            )

            # Reset the consumed_samples_total and epoch_consumed_samples for resuming dataloder
            # Do not use `train_steps`, since each timestep in an update step the same sample
            consumed_steps = client_state[0]['update_steps']
            self.ss.consumed_samples_total = defaultdict(int)
            self.ss.epoch_consumed_samples = defaultdict(int)
            for key in self.all_dataset_keys:
                self.ss.consumed_samples_total[key] = int(self.args.micro_batch_size * consumed_steps * self.dp_size / self.num_generations)
                self.ss.epoch_consumed_samples[key] = int(self.args.micro_batch_size * consumed_steps * self.dp_size / self.num_generations)
            self.logger.info(f"Resumed ScalarStates: {self.ss}")


    def build_reference_model(self):
        """
        Build the reference model. Note that model must be in `.eval()` mode.
        """
        self.logger.info("Building reference model...")
        factor_kwargs = {"device": "cpu", "dtype": PRECISION_TO_TYPE[self.args.precision]}
        # assert self.args.get('pretrained_reference_model_ckpt', None) is not None, "pretrained_reference_model_ckpt is not set."
        self.ref_model, _ = build_model(self.args, self.args.get('pretrained_reference_model_ckpt', None), logger=self.logger, **factor_kwargs)
        self.ref_model.requires_grad = False
        # self.ref_model = self.ref_model.to(self.device)
        # self.ref_model.eval()

        if self.args.pp_size > 1 or self.args.ep_size > 1:
            from hymm.parallelism.engines.gemini_parallel import GeminiParallelEngine
            if (self.parallel_dims.ep or self.parallel_dims.pp) and self.args.get('pretrained_reference_model_ckpt', None):
                load_ckpt_path = self.args.get('pretrained_reference_model_ckpt', None)
            else:
                load_ckpt_path = None
            self.ref_model = GeminiParallelEngine(
                model=self.ref_model,
                load_ckpt_path=load_ckpt_path,
                micro_batch_size=1,
                pp_enable_autocast=self.args.autocast_dtype != 'fp32',
                # pp_enable_autocast=False,
                autocast_prec=self.args.autocast_dtype,
                weight_prec=self.args.precision,
                # cpu_offload=True,
            )
            self.ref_model.eval()

        self.logger.info(f"Reference model built on device: {self.device}")

        assert self.dp_rank_is_correct
        loguru.logger.info(f'when seeding, {self.dp_rank=}')
        set_manual_seed(self.args.global_seed + self.dp_rank)
    

    def prepare_model_grpo_t2i_inputs(self, batch: Dict, device: Union[int, str], samples: Optional[List[Dict]]=None, timesteps_train: List[int]=None, **kwargs):
        input_prompts = batch["text"]
        batch_size = len(input_prompts)
        n_tokens = sum([len(prompt) for prompt in input_prompts])
        # TODO: 目前只支持每个rank bs=1的情况
        if "reward_tags" in batch:
            reward_tags = batch["reward_tags"][0]
        else:
            reward_tags = None

        # ===== update sde_noise_scale =====
        if self.update_sde_noise_scale:
            if self.baseline_exploration is None and self.exploration_score is not None:
                import copy
                self.baseline_exploration = copy.deepcopy(self.exploration_score)
            if self.exploration_score is not None and self.baseline_exploration is not None:
                self.sde_noise_scale = adaptive_noise_scale(
                    current_exploration=self.exploration_score,
                    baseline_exploration=self.baseline_exploration,
                    initial_noise_scale=self.initial_noise_scale,
                    min_noise_scale=self.min_noise_scale,
                    max_noise_scale=self.max_noise_scale,
                )
                self.logger.info(f"Updated sde_noise_scale: {self.sde_noise_scale}")

        ################################ 1. sample images (deepspeed) ################################
        self.model_engine.eval()
        with torch.no_grad():
            with unwrap_model_for_generation_deepspeed(self.model_engine) as unwrapped_model:
                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                    self.sampler.pipeline.model = unwrapped_model
                    self.sampler.model_dict["model"] = unwrapped_model
                    
                    if self.same_x0:
                        # Set seed for each prompt, so as to ensure that generating samples using the same initial latents;
                        # Set `deterministic` to False for the timesteps that are trained, otherwise set to True;
                        # In this way, different samples are generated for the same prompt by performing SDE operations at trainable 
                        # intermediate timesteps. (The variance in sde is different for each rank due to the initial different seeds.)
                        seeds = batch["seeds"]
                    else:
                        seeds = None
                    if self.training_strategy in ["progressive", "random", "decay", "dynamic"]:
                        determistic = [True] * self.num_train_timesteps
                        for timestep_i in timesteps_train:
                            determistic[timestep_i] = False
                    else:
                        determistic = [False] * self.num_train_timesteps
                    
                    # NOTE: T2TI任务中需要先对于同样的input prompts, 设置同样的seed, 保证生文完全相同，
                    # 然后在生图阶段再通过explicit_set_manual_seed_for_image_gen来设置不同的seed, 保证生图的sde随机性
                    if self.gen_text_freeze_for_same_prompt:
                        # Enable deterministic mode for text generation to ensure all ranks generate the same text
                        # This will be automatically disabled in sampler before image generation
                        torch.use_deterministic_algorithms(True)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                        set_manual_seed(seeds[0])
                    
                    # return_ref_logits = self.loss_mode in ["text_only", "both"] and self.kl_weight_text > 0
                    out_dict = self.sampler.batch_x2image(
                        batch_prompt_list=[input_prompts],
                        batch_system_prompt=[self.args.t2i_system_prompt] if self.args.t2i_system_prompt else None,
                        seed=seeds,
                        verbose=1,
                        task="t2i",
                        text_gen_seed=None,  # text_gen_seed is only used in batch_x2text, to generate different text sequences of each dp_rank with the same prompt.
                        sequence_template=self.args.sequence_template,
                        predict_image_shape_token=self.args.predict_image_shape_token,
                        sample_image_size=self.args.sample_image_size,
                        return_only_samples=False,
                        grpo_gen_text_mask_type="recaption",
                        explicit_set_manual_seed_for_image_gen={"seed": self.args.global_seed + self.dp_rank} if self.gen_text_freeze_for_same_prompt else None,
                        stop_by_all_rank=True, # Used for expert parallel: if stop_by_all_rank is True, the inference will stop when all ranks finished. 
                        max_new_tokens=self.args.cot_max_length,
                        pipeline_kwargs={
                            "kl_weight": self.kl_weight,
                            "determistic": determistic,
                            "sde_noise_scale": self.sde_noise_scale,
                            # "text_kl_weight": self.kl_weight_text if return_ref_logits else 0.0,
                        },
                    )
        self.model_engine.train()

        gen_imgs = out_dict["samples"]["samples"]  # list of PIL.Image.Image
        # if return_ref_logits:
        # 目前还不支持kv-cache返回ref logits，只支持不开kv-cache的情况
        #     all_latents, all_log_probs, all_ref_prev_latents_mean, model_input_extra_kwargs, text_ref_logits, text_mask_dict = out_dict["extra_outputs"]
        #     text_ref_logits = text_ref_logits[:1, :-1].clone().detach() # [1, sequence_length-1, vocab_size]
        # else:
        all_latents, all_log_probs, all_ref_prev_latents_mean, model_input_extra_kwargs, text_mask_dict = out_dict["extra_outputs"]
        text_ref_logits = None

        # Convert inference tensors to normal tensors that can be used in autograd, we need .clone() to create new tensors from inference tensors
        all_latents = [latent.clone().detach() for latent in all_latents]
        all_log_probs = [log_prob.clone().detach() for log_prob in all_log_probs]
        all_ref_prev_latents_mean = [ref_prev_latents_mean.clone().detach() for ref_prev_latents_mean in all_ref_prev_latents_mean]
        model_input_extra_kwargs = {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in model_input_extra_kwargs.items()}
        model_input_extra_kwargs["return_loss"] = False
        text_loss_mask = text_mask_dict['text_mask'][:1, 1:].bool() # [1, sequence_length-1]

        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 16, 96, 96)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps)
        if self.kl_weight > 0:
            all_ref_prev_latents_mean = torch.stack(all_ref_prev_latents_mean, dim=1)  # (batch_size, num_steps, ...)
        else:
            all_ref_prev_latents_mean = None
        timesteps = self.sampler.pipeline.scheduler.timesteps.repeat(
            len(input_prompts), 1
        )  # (batch_size, num_steps)

        ################################ 2. compute rewards ################################
        if "short_caption" in batch:
            # hps/ir/ps reward models are based on clip text features, which demand token length of input prompts less than 77
            reward_caption = batch['short_caption']
        else:
            reward_caption = input_prompts
        rewards, successes, rewards_dict, successes_dict = self.compute_reward(gen_imgs, reward_caption, reward_tags)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        successes = torch.from_numpy(np.array(successes)).int().to(self.device)
        rewards_dict = {k: torch.from_numpy(np.array(v)).float() for k, v in rewards_dict.items()}
        rewards_dict = dict(sorted(rewards_dict.items()))
        successes_dict = {k: torch.from_numpy(np.array(v)).int() for k, v in successes_dict.items()}

        if samples is None:
            samples = []
        samples.append(
            {
                "prompts": input_prompts,
                "timesteps": timesteps,
                "latents": all_latents[:, :-1], # each entry is the latent before timestep t
                "next_latents": all_latents[:, 1:], # each entry is the latent after timestep t
                "log_probs": all_log_probs,
                "ref_prev_latents_mean": all_ref_prev_latents_mean,
                "rewards": rewards,
                "successes": successes,
                "model_input_extra_kwargs": model_input_extra_kwargs,
                "rewards_dict": rewards_dict,
                "successes_dict": successes_dict,
                "text_loss_mask": text_loss_mask,
                "text_ref_logits": text_ref_logits,
            }
        )

        ################################ 3. compute advantages ################################
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        gathered_rewards = gather_tensor(samples[0]["rewards"]).view(-1)  # [world_size, batch_size]
        gathered_successes = gather_tensor(samples[0]["successes"]).view(-1)  # [world_size, batch_size]
        reward_mask = self.gen_reward_mask(gathered_successes)
        self.logger.info(f"gathered_rewards: {gathered_rewards}")
        self.logger.info(f"gathered_successes: {gathered_successes.bool()}")

        if self.training_obj == "advantage":
            # Get process slice for local data
            process_slice = slice(
                dist.get_rank() * len(samples[0]["prompts"]),
                (dist.get_rank() + 1) * len(samples[0]["prompts"]),
            )
            # Compute advantages using the specified normalization strategy
            advantages, avg_exploration_score = self.compute_advantages(
                gathered_rewards,
                reward_mask,
                self.num_generations,
                process_slice,
                compute_exploration_score=True,
            )
            samples[0]["advantages"] = advantages
            self.exploration_score = avg_exploration_score

            # Drop the advantages when too much False in group
            local_gathered_successes = reward_mask[int(dist.get_rank() / self.num_generations)]
            self.logger.info(f"local_gathered_successes: {local_gathered_successes}")
            if local_gathered_successes.sum().cpu().item() < (1.0-self.args.get("drop_false_percent", 0.0)) * self.num_generations:
                self.logger.warning(f"Too many False in gathered successes: {local_gathered_successes}")
                samples[0]["successes"] = torch.zeros_like(samples[0]["successes"])
                samples[0]["advantages"] = -self.adv_clip_max * torch.ones_like(advantages)
            else:
                samples[0]["advantages"] = advantages

        return samples, batch_size, n_tokens

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str], samples: Optional[List[Dict]]=None, timesteps_train: List[int]=None, **kwargs):
        if batch["dtype"][0] == "t2i":
            inputs = self.prepare_model_grpo_t2i_inputs(batch, device, samples, timesteps_train, **kwargs)
        else:
            raise ValueError(f"Unknown batch dtype, expected {self.all_dataset_keys}, got {batch['dtype']}")
        return inputs

    @property
    def optimizer(self):
        return self.model_engine.optimizer

    def update_reference_model(self, decay=0.95):
        """
        Update reference model using EMA from current training model.
        Args:
            decay: EMA decay rate, default 0.95
        """
        start_time = time.time()
        ref_model_params = OrderedDict(self.ref_model.named_parameters())
        current_model_params = OrderedDict(self.model_engine.named_parameters())
        
        for name, ref_param in ref_model_params.items():
            ref_param.data.mul_(decay).add_(current_model_params[name].data, alpha=1.0 - decay)
        end_time = time.time()
        self.logger.info(f"Updated reference model with EMA decay={decay} at step {self.ss.update_steps}, time cost: {end_time - start_time}s")
    
    def train_step(
            self,
            batch,
            timestep_i: int,
            cache_samples=None,
            cur_batch_size=None,
            n_tokens=None,
            timesteps_train: List[int]=None,
            only_kl_loss: bool=False,
    ):      
        # TODO: 支持 off-policy训练, 目前是纯on-policy训练, 没用到 importance sampling
        start1 = time.time()
        if cache_samples is None:
            # torch.cuda.empty_cache() # memory leak
            if self.ref_model_cpu_offload:
                self.ref_model.cuda()
            samples, cur_batch_size, n_tokens = self.prepare_model_inputs(batch, self.device, timesteps_train=timesteps_train)
            # torch.cuda.empty_cache() # memory leak
            if self.ref_model_cpu_offload:
                self.ref_model.cpu()
        else:
            if self.ref_model_cpu_offload:
                self.ref_model.cpu()
            samples = cache_samples
            assert (
                cur_batch_size is not None and n_tokens is not None
            ), "`cur_batch_size` and `n_tokens` must be provided if `cache_samples` is not None"

        torch.cuda.synchronize()
        duration1 = time.time() - start1

        start2 = time.time()

        def loss_closure(model_out, moe_loss=None):
            pred = model_out["diffusion_prediction"]
            pipeline = self.sampler.pipeline
            sample = samples[0]
            step_i = timestep_i
            from hymm.ar.pipelines.pipeline_transfusion_text2image_with_logprob import rescale_noise_cfg
            from hymm.diffusion.pipelines.flow_sde_with_logprob import sde_step_with_logprob

            pred = pred.to(dtype=torch.float32)

            # perform guidance
            if pipeline.do_classifier_free_guidance and not pipeline.do_face_classifier_free_guidance:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + pipeline.guidance_scale * (pred_cond - pred_uncond)

            elif pipeline.do_classifier_free_guidance and pipeline.do_face_classifier_free_guidance:
                # Use the text,image cfg  Equation 3 in https://arxiv.org/abs/2211.09800
                pred_cond, pred_uncond_text, pred_uncond_text_uncond_face = pred.chunk(3)
                pred = pred_uncond_text_uncond_face + \
                       pipeline.guidance_scale *       (pred_cond - pred_uncond_text) + \
                       pipeline.face_guidance_scale *  (pred_uncond_text - pred_uncond_text_uncond_face)

            if pipeline.do_classifier_free_guidance and pipeline.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=pipeline.guidance_rescale)

            # compute the log prob of next_latents given latents under the current model
            prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
                pipeline.scheduler,
                pred.float(),
                sample["timesteps"][:, step_i],
                sample["latents"][:, step_i].float(),
                prev_sample=sample["next_latents"][:, step_i].float(),
                determistic=False,
                sde_noise_scale=self.sde_noise_scale,
            )

            if self.kl_weight > 0:
                prev_sample_mean_ref = samples[0]["ref_prev_latents_mean"][:, timestep_i]
            else:
                prev_sample_mean_ref = None
            
            if self.loss_mode in ['text_only', 'both']:
                text_logits = model_out["logits"] # [2, sequence_length, vocab_size]
                text_logits = text_logits[:1, :-1]
            else:
                text_logits = None

            # grpo logic
            loss, policy_loss, kl_loss = compute_multimodal_grpo_loss(
                advantages=samples[0]["advantages"],
                log_prob=log_prob,
                old_log_probs=samples[0]["log_probs"][:, timestep_i],
                adv_clip_max=self.adv_clip_max,
                clip_range=self.clip_range,
                kl_weight=self.kl_weight,
                prev_sample_mean=prev_sample_mean,
                prev_sample_mean_ref=prev_sample_mean_ref,
                std_dev_t=std_dev_t,
                text_loss_mask=samples[0]["text_loss_mask"],
                text_logits=text_logits,
                text_ref_logits=samples[0]["text_ref_logits"],
                kl_weight_text=self.kl_weight_text,
                loss_mode=self.loss_mode,
                only_kl_loss=only_kl_loss,
            )

            if moe_loss is not None:
                loss = loss + moe_loss * self.args.get('moe_loss_weight', 0.01)

            # TODO: 目前只支持每个rank bs=1的情况：如果batch中没有成功的样本，则直接返回0损失；
            # 不能直接返回tensor(0.0)，因为ds需要保证计算图的完整性
            if samples[0]["successes"].sum().cpu().item() == 0:
                loss = loss * 0.0

            return loss, {
                'kl_loss': kl_loss.detach(),
                'policy_loss': policy_loss.detach(),
                'loss': loss.detach(),
                # 'moe_loss': moe_loss.detach() if moe_loss is not None else None,
            }

        self.model_engine.register_loss_closure(loss_closure)

        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            pipeline = self.sampler.pipeline
            sample = samples[0]
            step_i = timestep_i

            cfg_factor = 1
            if pipeline.do_classifier_free_guidance and not pipeline.do_face_classifier_free_guidance:
                cfg_factor = 2
            elif pipeline.do_classifier_free_guidance and pipeline.do_face_classifier_free_guidance:
                cfg_factor = 3
            elif pipeline.do_classifier_free_guidance is False and pipeline.do_face_classifier_free_guidance is True:
                raise NotImplementedError("Face guidance is not supported without classifier free guidance")

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([sample["latents"][:, step_i]] * cfg_factor)
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, sample["timesteps"][:, step_i])
            t_expand = sample["timesteps"][:, step_i].repeat(latent_model_input.shape[0])

            model_out = self.model_engine(
                x_t=latent_model_input,
                t=t_expand,
                **samples[0]["model_input_extra_kwargs"],
            )
            if self.parallel_dims.pp_enabled:   
                loss_dict = self.model_engine.get_cached_result('loss_dict', merge_op='mean')
                loss = loss_dict['loss']
            else:
                loss, loss_dict = loss_closure(model_out)
            
            kl_loss = loss_dict["kl_loss"]
            policy_loss = loss_dict["policy_loss"]
            moe_loss = None #loss_dict['moe_loss']
        
        exploration_dummyloss = torch.tensor(self.exploration_score, device=self.device, dtype=torch.float)
        loss_dict = {
            "loss": loss,
            "sde_noise_dummyloss": torch.tensor(self.sde_noise_scale, device=self.device, dtype=torch.float),
            "exploration_dummyloss": exploration_dummyloss,
            "reward_dummyloss": samples[0]["rewards"],
            "kl_loss": kl_loss,
            "policy_loss": policy_loss,
        }
        if moe_loss is not None:
            loss_dict["moe_loss"] = moe_loss

        # 多个rewards model训练时，用来记录当前rank每个reward model是否用到了，便于打 log
        all_rewards_exist_dict = {}
        
        # Add all reward losses with dummy values first
        for k in self.all_reward_loss_keys:
            loss_dict[k] = torch.zeros_like(loss, dtype=torch.float)
            # Convert tensor to list for easier gathering
            all_rewards_exist_dict[k] = False
        
        if self.training_obj == "advantage":
            loss_dict["advantages_dummyloss"] = samples[0]["advantages"]

        # Update reward dict with actual values
        for k, v in samples[0]["rewards_dict"].items():
            dummy_k = f"{k}_dummyloss"
            if dummy_k in self.all_reward_loss_keys:
                loss_dict[dummy_k] = v
                # Convert tensor to list for easier gathering
                all_rewards_exist_dict[dummy_k] = True

        torch.cuda.synchronize()
        duration2 = time.time() - start2

        times = {
            "preprocess": duration1,
            "forward": duration2,
        }

        return loss_dict, cur_batch_size, n_tokens, times, samples, all_rewards_exist_dict

    def update_train_states(self, ss, cs, batch, batch_size, n_tokens, loss):
        # A forward-backward step is counted as one train step.
        ss.add(train_steps=1, epoch_train_steps=1)
        cs.add(log_steps=1, running_loss=loss)
        # If training long sequence, each sequence may contain multiple samples.
        # Therefore, we sum `n_samples` to get the real number of samples.
        samples = batch["n_samples"].sum().item()
        key = batch["dtype"][0]
        cs.running_samples[key] += samples
        cs.running_tokens[key] += batch_size * n_tokens

        # We enable `is_update_step` if the current step is the gradient accumulation boundary.
        is_update_step = self.ss.train_steps % self.grad_accu_steps == 0
        if is_update_step:
            ss.add(
                update_steps=1,
                epoch_update_steps=1,
                current_run_update_steps=1
            )
            ss.lr = self.optimizer.param_groups[0]["lr"]

        return is_update_step
    
    def update_log_states(self, ss, all_cs, grad_accu_steps, num_train_timesteps):
        assert (
            grad_accu_steps % num_train_timesteps == 0
        ), "grad_accu_steps must be divisible by num_train_timesteps, otherwise the `consumed_samples` computation will be incorrect"
        consumed_samples = 0
        consumed_tokens = 0
        dp_ratio = self.world_size // self.dp_size
        for key in self.all_dataset_keys:
            part_samples = int(sum([cs_i.running_samples[key] for cs_i in all_cs]) / (num_train_timesteps * self.num_generations * dp_ratio))
            consumed_samples += part_samples
            self.ss.consumed_samples_total[key] += part_samples
            self.ss.epoch_consumed_samples[key] += part_samples
            part_tokens = int(sum([cs_i.running_tokens[key] for cs_i in all_cs]) / (dp_ratio))
            consumed_tokens += part_tokens
            self.ss.consumed_tokens_total[key] += part_tokens

        self.ss.add(
            consumed_computations_attn=6 * self.params_count["attn+mlp"] * consumed_tokens / C_SCALE,
            consumed_computations_total=6 * self.params_count["total"] * consumed_tokens / C_SCALE,
        )

        return consumed_samples

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
        if self.training_strategy in ["progressive", "random", "decay", "dynamic"]:
            # Initialize grpo training states
            self.grpo_states = GRPOTrainingStates(
                cur_timestep=self.mixgrpo_cur_timesteps,
                iters_per_group=self.train_iters_per_timesteps_group,
                group_size=self.timesteps_group_size,
                max_timesteps=self.num_train_timesteps,
                sample_strategy=self.training_strategy,
                overlap=self.timesteps_group_overlap,
                stride=self.mixgrpo_stride,
            )
            if self.training_strategy == "decay":
                self.grpo_states.set_params(self.decay_kwargs)
            elif self.training_strategy == "dynamic":
                self.grpo_states.set_params(self.dynamic_kwargs)

            # resume grpo states
            if self.resume_puretorch:
                self.grpo_states.restore_from_scalar_states(self.ss)

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

                    if self.args.get('save_last_batch_text', False):
                        text_batch = batch['text']
                        gather_text_list = [None for _ in range(dist.get_world_size())]
                        torch.distributed.all_gather_object(gather_text_list, text_batch)
                        if self.rank == 0:
                            import pickle
                            with open(os.path.join(self.exp_dir, f"text_last_batch.pkl"), "wb") as f:
                                pickle.dump(gather_text_list, f)

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
                    
                    samples = None
                    batch_size = None
                    n_tokens = None
                    if self.training_strategy == "all":
                        timesteps_train = [ti for ti in range(self.num_train_timesteps)]
                    elif self.training_strategy in ["progressive", "random", "decay", "dynamic"]:
                        timesteps_train = self.grpo_states.get_current_timesteps()
                        only_kl_loss = False
                        self.grpo_states.update_iteration(seed=batch["seeds"][0] if self.training_strategy == "random" else None)

                    if self.use_extra_low_timesteps_kl:
                        extra_kl_start_timestep = self.args.get("extra_kl_start_timestep", 20)
                        extra_kl_end_timestep = self.args.get("extra_kl_end_timestep", 29)
                        timesteps_train = timesteps_train + [
                            random.randint(
                                extra_kl_start_timestep,
                                extra_kl_end_timestep,
                            )
                        ]

                    for timestep_idx in timesteps_train:
                        if self.use_extra_low_timesteps_kl and timestep_idx >= extra_kl_start_timestep and timestep_idx <= extra_kl_end_timestep:
                            only_kl_loss = True
                        else:
                            only_kl_loss = False

                        (
                            loss_dict,
                            batch_size,
                            n_tokens,
                            forward_times,
                            samples,
                            all_rewards_exist_dict
                        ) = self.train_step(
                            batch,
                            timestep_idx,
                            samples,
                            batch_size,
                            n_tokens,
                            timesteps_train,
                            only_kl_loss,
                        )
                        self.logger.info(f"rank-{self.rank}, timestep-{timestep_idx}, loss: {loss_dict['loss']}")
                        times.update(forward_times)

                        backward_start = time.time()
                        loss = loss_dict["loss"].mean()
                        for k, v in loss_dict.items():
                            if "loss" in k and k != "loss":
                                cs.running_sub_loss_dict[k] += v.mean().item()
                                # Only update loss if it exists in current rank
                                # TODO: 仅支持batchsize=1的情况
                                if k not in all_rewards_exist_dict or (k in all_rewards_exist_dict and all_rewards_exist_dict[k]):
                                    # cs.running_sub_loss_dict[k] += v.mean().item()
                                    cs.running_sub_step_dict[k] += 1
                        self.model_engine.backward(loss)
                        torch.cuda.synchronize()
                        times['backward'] = time.time() - backward_start
                        is_update_step = self.update_train_states(self.ss, cs, batch, batch_size, n_tokens, loss.item())

                        # Update model parameters at the boundary of gradient accumulation.
                        update_start = time.time()
                        # Get the lr before optimizer.step()
                        lrs = [group["lr"] for group in self.model_engine.optimizer_container.param_groups]
                        self.model_engine.step()
                        torch.cuda.synchronize()
                        times['update'] = time.time() - update_start

                        if self.ss.update_steps >= args.max_training_steps:
                            # Enter stopping routine if max steps reached after this step.
                            finished = True

                        # Update EMA model at the step of main model parameters update.
                        if args.use_ema and is_update_step:
                            self.ema.update(self.model_engine.module)

                        # Update reference model
                        if is_update_step and self.update_ref_model and self.ss.update_steps % self.ema_update_ref_steps == 0:
                            self.ref_model.reshard()
                            self.update_reference_model()

                        # Log training information:
                        if is_update_step and self.ss.update_steps % args.log_every == 0:
                            # All-gather scalar states and cycle states.
                            all_cs: List[Optional[CycleStates]] = [None for _ in range(self.world_size)]
                            torch.distributed.all_gather_object(all_cs, cs)
                            all_rewards_exist_dicts: List[Optional[Dict[str, bool]]] = [None for _ in range(self.world_size)]
                            torch.distributed.all_gather_object(all_rewards_exist_dicts, all_rewards_exist_dict)

                            # Calculate average main loss
                            avg_loss = sum([cs_i.running_loss for cs_i in all_cs]) / sum([cs_i.log_steps for cs_i in all_cs])
                            
                            # Calculate average sub losses based on exist flags
                            merged_loss_dict = {}
                            merged_step_dict = {}
                            for k in sorted(cs.running_sub_loss_dict.keys()):
                                total_loss = 0
                                total_steps = 0
                                exist_count = 0
                                for rank_idx, (cs_i, exist_dict) in enumerate(zip(all_cs, all_rewards_exist_dicts)):
                                    if k in cs_i.running_sub_loss_dict:
                                        # For reward losses, check exist flag; for other losses, always include
                                        should_include = True
                                        if k in exist_dict:
                                            should_include = exist_dict[k]
                                        
                                        if should_include:
                                            total_loss += cs_i.running_sub_loss_dict[k]
                                            total_steps += cs_i.running_sub_step_dict[k]
                                            exist_count += 1
                                
                                if exist_count > 0:  # Only include if at least one rank has this loss
                                    merged_loss_dict[k] = total_loss
                                    merged_step_dict[k] = total_steps
                            
                            avg_sub_loss_dict = {k: merged_loss_dict[k] / merged_step_dict[k] for k in merged_loss_dict}
                            # Calculate cumulated metrics.
                            cum_samples = self.update_log_states(self.ss, all_cs, self.grad_accu_steps, len(timesteps_train))

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
                        # torch.cuda.empty_cache()

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


    def build_dataloader(self):
        args = self.args
        self.dataloader_preliminary_setup()

        dataloader_kwargs = dict(
            **args.dataloader_params,
            worker_init_fn=set_worker_seed_builder(self.dp_rank),
            shuffle=False,
            drop_last=True,
        )
        sampler_kwargs = dict(
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
        )

        def _filter_dummies(dummy_list):
            # Filter out dummies that are not in the dataset keys
            valid_dummies = []
            for dummy_candidate in dummy_list:
                if any(task in self.all_dataset_keys for task in self.dummy_to_tasks[dummy_candidate]):
                    valid_dummies.append(dummy_candidate)
            return valid_dummies

        # =====================================
        #     Text(+Image) to image data
        # =====================================
        self.task_dummy_dict['t2i'] = _filter_dummies(['mmu'])
        self.task_dummy_dict['face_id_clip'] = []
        task_info_list = [
            dict(dataset_tag="t2i", cur_task="t2i", cls=RLTextImageArrowStream),
        ]
        for item in task_info_list:
            dataset_tag = item['dataset_tag']
            if dataset_tag not in self.all_dataset_keys:
                continue
            task_batch_size = args.get(f'{dataset_tag}_batch_size', self.micro_batch_size)

            if dataset_tag == "t2i":
                multireso = args["t2i_index_kwargs"]["multireso"]
            else:
                multireso = args.get(f'{dataset_tag}_index_kwargs')[f"{dataset_tag}_multireso"]

            assert self.dp_rank_is_correct
            assert self.dp_size * (task_batch_size if multireso else 1) % self.num_generations == 0, f"Gpu is not sufficient with dp_size:{self.dp_size}. Consider reducing num_generations / increasing gpu numbers / reducing PP."

            self.dataset_dict[dataset_tag] = item['cls'](
                args=args,
                dataset_tag=dataset_tag,
                tokenizer_name=args.tokenizer_name,
                task_kwargs=args.get(f'{dataset_tag}_task_kwargs'),
                index_kwargs=dict(
                    batch_size=task_batch_size if multireso else 1,  # Provide bsz to use multireso.
                    world_size=1,  # Dataset don't need to align with world_size. It will be handled by the sampler.
                    **args.get(f'{dataset_tag}_index_kwargs'),
                ),
                logger=self.logger,
                dummy_number=sum([self.dummy_dict[task] for task in self.task_dummy_dict[item['cur_task']]]),
                template=args.sequence_template,
                # if sequence batch is enabled, attention mask sequence length -1 is disabled in __getitem__,
                # and performed in seq_collate_fn instead.
                attn_mask_seq_m1=dataset_tag not in self.seq_batch_size,
            )
            # Build repeated sampler and data loader
            self.sampler_dict[dataset_tag] = RepeatRandomDistributedSampler(
                self.dataset_dict[dataset_tag],
                num_replicas=self.dataset_num_replicas[dataset_tag],
                rank=self.dataset_rank[dataset_tag],
                batch_size=task_batch_size if multireso else 1,  # Provide bsz to use multireso.
                mini_repeat_count=self.num_generations,
                repeat_count=self.num_grpo_iterations,
                **sampler_kwargs,
            )
            self.dataloader_dict[dataset_tag] = DataLoader(
                self.dataset_dict[dataset_tag],
                batch_size=task_batch_size,
                sampler=self.sampler_dict[dataset_tag],
                collate_fn=(self.dataset_dict[dataset_tag].collate_fn
                            if hasattr(self.dataset_dict[dataset_tag], "collate_fn")
                            else None),
                **dataloader_kwargs,
            )

    def save_checkpoint(self):
        # Update GRPO states in scalar states before saving
        if hasattr(self, 'grpo_states'):
            self.grpo_states.update_scalar_states(self.ss)

        save_checkpoint(self.args, self.rank, self.logger, self.model_engine, self.ema, self.ss, self.ckpt_dir)

    def get_states_cls(self, state_type):
        if state_type == "scalar":
            from hymm.trainers.helpers import MultiModalGRPOScalarStates
            return MultiModalGRPOScalarStates
        elif state_type == "cycle":
            from hymm.trainers.helpers import MultiModalCycleStates
            return MultiModalCycleStates
        else:
            raise ValueError(f"Unknown state type: {state_type}")
