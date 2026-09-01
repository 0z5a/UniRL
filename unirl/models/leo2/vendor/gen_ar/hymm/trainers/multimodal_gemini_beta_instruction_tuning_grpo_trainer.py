import gc
import os
import time
import json
from typing import Dict, Union, List, Optional
import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ..ar.pipelines.pipeline_transfusion_text2image_with_logprob import compute_log_prob
from ..data_kits.combined_iterator import CombinedBatchIterator
from ..data_kits.rl_t2i_loader import RLTextImageArrowStream
from ..data_kits.samplers import RepeatRandomDistributedSampler
from .helpers import (
    CycleStates,
    save_checkpoint,
)
from ..models import build_model
from ..models.reward_models.text_ocr_groundingQwenVL import TextOCRGroundingQwenVL
from .multimodal_gemini_beta_grpo_trainer import MultiModalGeminiBetaGRPOTrainer
from ..samplers.gemini_beta_sampler import GeminiBetaSampler
from ..utils.torch_utils import (
    set_manual_seed,
    set_worker_seed_builder,
    PRECISION_TO_TYPE,
)
from ..utils.deepspeed_utils import unwrap_model_for_generation_deepspeed
from ..utils.torch_utils import move_model_params_and_grads_to
from ..utils.torch_distributions import gather_tensor
from ..utils.torch_utils import profiler_context
from .helpers import GRPOTrainingStates
from ..models.reward_models.google_gemini_request import GoogleGeminiRewardModel, SubDriCons_RM_Face, SubDriCons_RM_Gemini, SemAlign_RM
from ..models.reward_models.hps_clip import HPSClipRewardModel
from ..models.reward_models.unified_reward import UnifiedRewardModel
from ..models.reward_models.image_reward import ImageRewardModel
from ..models.reward_models.pick_score import PickScoreRewardModel

gc.set_threshold(7000, 100, 100)


class MultiModalGeminiBetaInstructionTuningGRPOTrainer(MultiModalGeminiBetaGRPOTrainer):
    def __init__(self, args, all_dataset_keys=None):
        super().__init__(args, all_dataset_keys)
        # 访问cos_url时，需要关掉代理
        self.unset_proxy()

        # Google Gemini Editing reward model: Use the remote server to compute the reward
        if self.args.get("google_gemini_editing", False):
            google_gemini_app_ids = self.args["google_gemini_editing"]["google_gemini_editing_app_id"]
            google_gemini_app_keys = self.args["google_gemini_editing"]["google_gemini_editing_app_key"]
            if isinstance(google_gemini_app_ids, list):
                assert (
                    len(google_gemini_app_ids) == len(google_gemini_app_keys)
                ), "The number of google_gemini_app_ids and google_gemini_app_keys must be the same"
                num_app_ids = len(google_gemini_app_ids)
                app_id_idx = self.rank % num_app_ids
                google_gemini_app_id = google_gemini_app_ids[app_id_idx]
                google_gemini_app_key = google_gemini_app_keys[app_id_idx]
            else:
                google_gemini_app_id = google_gemini_app_ids
                google_gemini_app_key = google_gemini_app_keys
                
            self.gemini_editing_reward_model = GoogleGeminiRewardModel(
                app_id=google_gemini_app_id,
                app_key=google_gemini_app_key,
                task_name=self.args["google_gemini_editing"].get("google_gemini_editing_task_name", "Editing"),
                logger=self.logger,
                score_weights=self.args["google_gemini_editing"].get("google_gemini_editing_score_weights", None),
                model_marker=self.args["google_gemini_editing"].get("google_gemini_editing_model_marker", None),
            )
            self.reward_models.append(self.gemini_editing_reward_model)
        
        self.reward_weights["GoogleGeminiRewardModel"] = self.args.get('google_gemini_editing_weight', 1.0)

        # Subject Driven reward model: Use the remote server to compute the reward
        if self.args.get("subject_driven_reward", False):
            google_gemini_app_ids = self.args["subject_driven_reward"]["google_gemini_subject_driven_app_id"]
            google_gemini_app_keys = self.args["subject_driven_reward"]["google_gemini_subject_driven_app_key"]
            if isinstance(google_gemini_app_ids, list):
                assert (
                    len(google_gemini_app_ids) == len(google_gemini_app_keys)
                ), "The number of google_gemini_app_ids and google_gemini_app_keys must be the same"
                num_app_ids = len(google_gemini_app_ids)
                app_id_idx = self.rank % num_app_ids
                google_gemini_app_id = google_gemini_app_ids[app_id_idx]
                google_gemini_app_key = google_gemini_app_keys[app_id_idx]
            else:
                google_gemini_app_id = google_gemini_app_ids
                google_gemini_app_key = google_gemini_app_keys
            
            if self.args["subject_driven_reward"].get("task_name", False):
                if self.args["subject_driven_reward"]["task_name"].get("consistency_face_weight", 0.0) > 0:
                    self.subject_driven_face_consistency_reward_model = SubDriCons_RM_Face(
                        http_proxy=self.args["subject_driven_reward"].get("face_http_proxy", None),
                        https_proxy=self.args["subject_driven_reward"].get("face_https_proxy", None),
                    )
                    self.reward_models.append(self.subject_driven_face_consistency_reward_model)
                    self.reward_weights["SubDriCons_RM_Face"] = self.args["subject_driven_reward"]["task_name"].get("consistency_face_weight", 0.0)

                if self.args["subject_driven_reward"]["task_name"].get("consistency_gemini_weight", 0.0) > 0:
                    self.subject_driven_consistency_reward_model = SubDriCons_RM_Gemini(
                        app_id=google_gemini_app_id,
                        app_key=google_gemini_app_key,
                        logger=self.logger,
                        model_marker=self.args["subject_driven_reward"].get("google_gemini_subject_driven_model_marker", None),
                    )
                    self.reward_models.append(self.subject_driven_consistency_reward_model)
                    self.reward_weights["SubDriCons_RM_Gemini"] = self.args["subject_driven_reward"]["task_name"].get("consistency_gemini_weight", 0.0)

                if self.args["subject_driven_reward"]["task_name"].get("semantic_alignment_weight", 0.0) > 0:
                    self.subject_driven_semantic_reward_model = SemAlign_RM(
                        app_id=google_gemini_app_id,
                        app_key=google_gemini_app_key,
                        logger=self.logger,
                        model_marker=self.args["subject_driven_reward"].get("google_gemini_subject_driven_model_marker", None),
                    )
                    self.reward_models.append(self.subject_driven_semantic_reward_model)
                    self.reward_weights["SemAlign_RM"] = self.args["subject_driven_reward"]["task_name"].get("semantic_alignment_weight", 0.0)

        # Initialize reward model weights only for activated models
        # Initialize all possible reward loss keys
        self.reward_weights = {}
        self.all_reward_loss_keys = []
        for model in self.reward_models:
            model_name = type(model).__name__
            self.all_reward_loss_keys.append(f"{model_name}_dummyloss")
            if model_name == 'TextOCRGroundingQwenVL':
                weight = self.args.get('text_ocr_weight', 1.0)
            elif model_name == 'GoogleGeminiRewardModel':
                weight = self.args.get('google_gemini_count_weight', 1.0)
            elif model_name == 'HPSClipRewardModel':
                weight = self.args.get('hps_clip_weight', 1.0)
            elif model_name == 'ImageRewardModel':
                weight = self.args.get('image_reward_weight', 1.0)
            elif model_name == 'UnifiedRewardModel':
                weight = self.args.get('unified_reward_weight', 1.0)
            elif model_name == 'PickScoreRewardModel':
                weight = self.args.get('pick_score_weight', 1.0)
            elif model_name == 'SubDriCons_RM_Face':
                weight = self.args["subject_driven_reward"]["task_name"].get("consistency_face_weight", 1.0)
            elif model_name == 'SubDriCons_RM_Gemini':
                weight = self.args["subject_driven_reward"]["task_name"].get("consistency_gemini_weight", 1.0)
            elif model_name == 'SubDriSem_RM':
                weight = self.args["subject_driven_reward"]["task_name"].get("semantic_alignment_weight", 1.0)
            else:
                weight = 1.0
            self.reward_weights[model_name] = weight

        # Normalize weights
        total_weight = sum(self.reward_weights.values())
        if total_weight > 0:
            self.reward_weights = {k: v/total_weight for k, v in self.reward_weights.items()}
        else:
            self.logger.warning("No reward models activated or all weights are 0!")
            self.reward_weights = {type(model).__name__: 1.0/len(self.reward_models) for model in self.reward_models}

    def prepare_model_grpo_ti2i_inputs(self, batch: Dict, device: Union[int, str], samples: Optional[List[Dict]]=None, timesteps_train: List[int]=None, **kwargs):
        input_prompts = batch["text"]
        pil_src_images = batch["src_images"]
        use_face_reward_flags = batch["use_face_rewards"]
        sem_points = batch["sem_points"]
        batch_size = len(input_prompts)
        n_tokens = sum([len(prompt) for prompt in input_prompts])
        # TODO: 目前只支持每个rank bs=1的情况
        if "reward_tags" in batch:
            reward_tags = batch["reward_tags"][0]
        else:
            reward_tags = None

        ################################ 1. sample images (deepspeed) ################################
        self.model_engine.module.eval()
        with torch.inference_mode():
            with unwrap_model_for_generation_deepspeed(self.model_engine) as unwrapped_model:
                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                    self.sampler.pipeline.model = unwrapped_model
                    
                    if self.training_strategy in ["progressive", "random", "decay", "dynamic"]:
                        # Set seed for each prompt, so as to ensure that generating samples using the same initial latents;
                        # Set `deterministic` to False for the timesteps that are trained, otherwise set to True;
                        # In this way, different samples are generated for the same prompt by performing SDE operations at trainable 
                        # intermediate timesteps. (The variance in sde is different for each rank due to the initial different seeds.)
                        seeds = batch["seeds"]
                        determistic = [True] * self.num_train_timesteps
                        for timestep_i in timesteps_train:
                            determistic[timestep_i] = False
                    else:
                        # seeds = None
                        # determistic = False
                        seeds = batch["seeds"]
                        determistic = [False] * self.num_train_timesteps

                    image_info = self.sampler.process_src_image(
                        pil_src_images[0],
                        dataset_base_size=self.args.training_image_size,
                        use_joint_image_feature=self.args.use_joint_image_feature,
                    )
                    if self.args.get("use_joint_feature_sampling", False):
                        out_dict = self.sampler.batch_x2image(
                            [input_prompts],
                            seed=seeds,
                            verbose=1,
                            task="ti2i",
                            sequence_template=self.args.sequence_template,
                            predict_image_shape_token=self.args.predict_image_shape_token,
                            sample_image_size=self.args.sample_image_size,
                            batch_joint_image_info_list=[[image_info]],
                            return_only_samples=False,
                            pipeline_kwargs={"kl_weight": self.kl_weight, "determistic": determistic},
                        )
                    else:
                        out_dict = self.sampler.batch_x2image(
                            [input_prompts],
                            seed=seeds,
                            verbose=1,
                            task="ti2i",
                            sequence_template=self.args.sequence_template,
                            predict_image_shape_token=self.args.predict_image_shape_token,
                            sample_image_size=self.args.sample_image_size,
                            batch_src_image_info_list=[[image_info]],
                            return_only_samples=False,
                            pipeline_kwargs={"kl_weight": self.kl_weight, "determistic": determistic},
                        )
        self.model_engine.module.train()

        gen_imgs = out_dict["samples"]["samples"]  # list of PIL.Image.Image
        all_latents, all_log_probs, all_ref_prev_latents_mean, model_input_extra_kwargs = out_dict["extra_outputs"]
        # Convert inference tensors to normal tensors that can be used in autograd, we need .clone() to create new tensors from inference tensors
        all_latents = [latent.clone().detach() for latent in all_latents]
        all_log_probs = [log_prob.clone().detach() for log_prob in all_log_probs]
        all_ref_prev_latents_mean = [ref_prev_latents_mean.clone().detach() for ref_prev_latents_mean in all_ref_prev_latents_mean]
        model_input_extra_kwargs = {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in model_input_extra_kwargs.items()}
        model_input_extra_kwargs["return_loss"] = False

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
        rewards, successes, rewards_dict, successes_dict = self.compute_reward(gen_imgs, input_prompts, reward_tags, src_images=pil_src_images, 
                                                                               use_face_rewards=use_face_reward_flags, sem_points=sem_points)
        
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        successes = torch.from_numpy(np.array(successes)).int().to(self.device)
        rewards_dict = {k: torch.from_numpy(np.array(v)).float() for k, v in rewards_dict.items()}
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
            advantages = self.compute_advantages(
                gathered_rewards,
                reward_mask,
                self.num_generations,
                process_slice,
            )
            # Drop the advantages when too much False in group
            local_gathered_successes = reward_mask[int(dist.get_rank() / self.num_generations)]
            self.logger.info(f"local_gathered_successes: {local_gathered_successes}")
            if local_gathered_successes.sum().cpu().item() < (1.0-self.args.get("drop_false_percent", 0.0)) * self.num_generations:
                self.logger.warning(f"Too many False in gathered successes: {local_gathered_successes}")
                samples[0]["successes"] = torch.zeros_like(samples[0]["successes"])

            samples[0]["advantages"] = advantages
            
        return samples, batch_size, n_tokens, 

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str], samples: Optional[List[Dict]]=None, timesteps_train: List[int]=None, **kwargs):
        if batch["dtype"][0] == "t2i":
            inputs = self.prepare_model_grpo_ti2i_inputs(batch, device, samples, timesteps_train, **kwargs)
        elif batch["dtype"][0] == "ti2i":
            inputs = self.prepare_model_grpo_ti2i_inputs(batch, device, samples, timesteps_train, **kwargs)
        else:
            raise ValueError(f"Unknown batch dtype, expected {self.all_dataset_keys}, got {batch['dtype']}")
        return inputs
