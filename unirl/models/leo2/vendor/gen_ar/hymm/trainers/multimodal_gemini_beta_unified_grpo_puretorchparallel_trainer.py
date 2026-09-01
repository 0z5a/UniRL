import gc
import os
import time
from typing import Dict, Union, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from hymm.data_kits.system_prompt import unified_system_prompt_en
from .multimodal_gemini_beta_t2ti_grpo_puretorchparallel_trainer import MultiModalGeminiT2TIBetaGRPOPureTorchParallelTrainer
from ..utils.torch_utils import set_manual_seed
from ..utils.deepspeed_utils import unwrap_model_for_generation_deepspeed
from ..utils.torch_distributions import gather_tensor
from ..utils.rl_exploration_metric import adaptive_noise_scale

gc.set_threshold(7000, 100, 100)


class MultiModalGeminiBetaUnifiedGRPOPureTorchParallelTrainer(MultiModalGeminiT2TIBetaGRPOPureTorchParallelTrainer):
    """
    Unified GRPO trainer for both x2i (t2i/ti2i) and x2ti (t2ti/ti2ti) tasks.
    """
    def __init__(self, args, all_dataset_keys=None):
        super().__init__(args)
        self.args.ti2i_system_prompt = unified_system_prompt_en
        # 通过enable_generate_text和cot_max_length参数来控制是否开启生文，如果开启生文的话，这里长度设置大于0，否则小于0
        self.enable_generate_text = self.args.get("enable_generate_text", False)
        if self.enable_generate_text:
            self.cot_max_length = self.args.get("cot_max_length", 768)
            self.args.cot_max_length = self.cot_max_length
        else:
            self.cot_max_length = self.args.cot_max_length = 0
    
    def task_init(self, args, all_dataset_keys=None):
        super().task_init(args, all_dataset_keys)

    def prepare_model_grpo_x2i_x2ti_inputs(self, batch: Dict, device: Union[int, str], samples: Optional[List[Dict]]=None, timesteps_train: List[int]=None, **kwargs):
        """ Support both x2i and x2ti 
        目前通过enable_generate_text和cot_max_length参数来控制是否开启是否先recaption再生成图片。
        TODO:
          - 支持通过yaml中设置数据采样概率来设置x2i和x2ti的比例，二者混合训练
          - 支持一个step中同时包含x2i和x2ti的情况，因为一个生成text，一个不生成，ep会有问题，需要把x2i做dummy generation
          - 支持放开vit encoder训练，这样的话需要对t2i/t2ti任务添加dummy token
        """

        input_prompts = batch["text"]
        pil_src_images = batch["src_images"]
        pil_ref_images = batch["ref_images"]
        task = batch['dtype'][0]
        use_face_reward_flags = batch["use_face_rewards"]
        sem_points = batch["sem_points"]
        subtask = batch["subtask"]

        batch_size = len(input_prompts)
        n_tokens = sum([len(prompt) for prompt in input_prompts])
        # we use task type (t2i, ti2i) as reward tags for computing rewards
        reward_tags = batch['dtype']
        # TODO: 目前只支持每个rank bs=1的情况
        # if "reward_tags" in batch:
        #     reward_tags = batch["reward_tags"][0]
        # else:
        #     reward_tags = None
        t_start = time.time()

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
                    
                    if pil_src_images is not None:
                        image_info = [[self.sampler.process_src_image(
                            pil,
                            dataset_base_size=self.args.training_image_size,
                            use_joint_image_feature=self.args.use_joint_image_feature,
                        ) for pil in pil_src_images]]
                    else:
                        image_info = None

                    # NOTE: T2TI任务中需要先对于同样的input prompts, 设置同样的seed, 保证生文完全相同，
                    # 然后在生图阶段再通过explicit_set_manual_seed_for_image_gen来设置不同的seed, 保证生图的sde随机性
                    if self.enable_generate_text and self.gen_text_freeze_for_same_prompt:
                        # Enable deterministic mode for text generation to ensure all ranks generate the same text
                        # This will be automatically disabled in sampler before image generation
                        torch.use_deterministic_algorithms(True)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                        set_manual_seed(seeds[0])

                    if self.args.get("use_joint_image_feature", False):
                        out_dict = self.sampler.batch_x2image(
                            [input_prompts],
                            batch_system_prompt=[self.args.ti2i_system_prompt] if self.args.ti2i_system_prompt else None,
                            seed=seeds,
                            verbose=1,
                            task=task,
                            sequence_template=self.args.sequence_template,
                            predict_image_shape_token=self.args.predict_image_shape_token,
                            sample_image_size=self.args.sample_image_size,
                            batch_joint_image_info_list=image_info,
                            text_gen_seed=None,
                            grpo_gen_text_mask_type="recaption",
                            explicit_set_manual_seed_for_image_gen={"seed": self.args.global_seed + self.dp_rank} if self.gen_text_freeze_for_same_prompt else None,
                            stop_by_all_rank=True, # Used for expert parallel: if stop_by_all_rank is True, the inference will stop when all ranks finished. 
                            max_new_tokens=self.cot_max_length,
                            return_only_samples=False,
                            pipeline_kwargs={
                                "kl_weight": self.kl_weight,
                                "determistic": determistic,
                                "sde_noise_scale": self.sde_noise_scale,
                                # "text_kl_weight": self.kl_weight_text if return_ref_logits else 0.0,
                            },
                        )
                    else:
                        out_dict = self.sampler.batch_x2image(
                            [input_prompts],
                            batch_system_prompt=[self.args.ti2i_system_prompt] if self.args.ti2i_system_prompt else None,
                            seed=seeds,
                            verbose=1,
                            task=task,
                            sequence_template=self.args.sequence_template,
                            predict_image_shape_token=self.args.predict_image_shape_token,
                            sample_image_size=self.args.sample_image_size,
                            batch_src_image_info_list=image_info,
                            text_gen_seed=None,
                            grpo_gen_text_mask_type="recaption",
                            explicit_set_manual_seed_for_image_gen={"seed": self.args.global_seed + self.dp_rank} if self.gen_text_freeze_for_same_prompt else None,
                            stop_by_all_rank=True, # Used for expert parallel: if stop_by_all_rank is True, the inference will stop when all ranks finished. 
                            max_new_tokens=self.cot_max_length,
                            return_only_samples=False,
                            pipeline_kwargs={
                                "kl_weight": self.kl_weight,
                                "determistic": determistic,
                                "sde_noise_scale": self.sde_noise_scale,
                                # "text_kl_weight": self.kl_weight_text if return_ref_logits else 0.0,
                            },
                        )
                        
        self.model_engine.train()
        t_rollout = time.time()

        gen_imgs = out_dict["samples"]["samples"]  # list of PIL.Image.Image
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
        rewards, successes, rewards_dict, successes_dict = self.compute_reward(
            gen_imgs, 
            reward_caption, 
            reward_tags,
            src_images=pil_src_images,
            use_face_rewards=use_face_reward_flags,
            sem_points=sem_points,
            subtask=subtask,
            ref_images=pil_ref_images,
        )

        t_reward = time.time()
        self.logger.info(f'Prepare input time cost: total={t_reward-t_start:.2f}s, rollout={t_rollout-t_start:.2f}s, reward={t_reward-t_rollout:.2f}s')

        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        successes = torch.from_numpy(np.array(successes)).int().to(self.device)
        rewards_dict = {k: torch.from_numpy(np.array(v)).float() for k, v in rewards_dict.items()}
        # 按照model_name进行排序，否则会因为并行的原因导致每个rank中model_name的顺序不一样，从而走advantage_aggr时gather_tensor时出错
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
        if self.multi_reward_mix == "advantage_aggr":
            gathered_rewards = {}
            for model_name, model_rewards in rewards_dict.items():
                gathered_rewards[model_name] = gather_tensor(model_rewards.to(self.device)).view(-1) # [world_size, batch_size]
                dist.barrier()
        elif self.multi_reward_mix == "reward_aggr":
            gathered_rewards = gather_tensor(samples[0]["rewards"]).view(-1) # [world_size, batch_size]
            dist.barrier()
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
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
        if batch["dtype"][0] in ["t2i", "ti2i"]:
            inputs = self.prepare_model_grpo_x2i_x2ti_inputs(batch, device, samples, timesteps_train, **kwargs)
        else:
            raise ValueError(f"Unknown batch dtype, expected {['t2i', 'ti2i']}, got {batch['dtype']}")
        return inputs
