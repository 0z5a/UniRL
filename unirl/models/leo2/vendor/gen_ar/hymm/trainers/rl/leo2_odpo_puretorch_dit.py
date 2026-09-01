import copy

import torch
import torch.distributed as dist
from hymm.trainers.rl.leo2_dpo_puretorch_dit import Leo2DPOTrainer
from hymm.trainers.rl.leo2_nft_puretorch_dit import Leo2NFTTrainer

class Leo2ODPOTTrainer(Leo2NFTTrainer):

    def compute_advantages(self, group_size, score_dict, reward_config):
        weighted_advantages = torch.zeros(group_size, device=self.device, dtype=torch.float32)
        total_weight = 0.0
        for model_name, model_config in reward_config["models"].items():
            model_weight = float(model_config["weight"])
            for metric_name, metric_weight in model_config["sub_reward"].items():
                metric_weight = float(metric_weight)
                if metric_weight == 0:
                    continue
                # Build the key: "{model_name}_{metric_name}" (lowercase)
                reward_key = f"{model_name}_{metric_name.lower()}"
                score_tensor = torch.tensor(score_dict[reward_key], dtype=torch.float32, device=self.device)
                # Normalize this reward separately
                normalized_advantage = (score_tensor - score_tensor.mean())/(score_tensor.std() + 1e-8)
                normalized_advantage = torch.nan_to_num(normalized_advantage, nan=0.0, posinf=0.0, neginf=0.0)

                # Weight and accumulate
                combined_weight = model_weight * metric_weight
                weighted_advantages = weighted_advantages + combined_weight * normalized_advantage
                total_weight += combined_weight

        # Final normalization (optional, can be skipped if relative weights are sufficient)
        weighted_advantages = (weighted_advantages - weighted_advantages.mean()) / (weighted_advantages.std() + 1e-8)
        weighted_advantages = 0.5 + 0.5 * torch.clamp(weighted_advantages, -1, 1)
        return weighted_advantages

    def prepare_micro_batches(self, micro_batches, batch):
        args = self.args
        dataset_tag = batch["dataset_tag"][0]
        task_kwargs = getattr(args, f"{dataset_tag}_task_kwargs")

        idx = batch["index"][0]
        prompt = batch["text"][0]
        seed = batch["seed"][0]
        ref_image_path = batch["ref_image_path"][0]

        #self.teacher_model_engine = update_model(self.model_engine, self.teacher_model_engine, decay=task_kwargs['teacher_update_decay'])
        # rollout samples
        group_size = int(task_kwargs['group_size'])
        with torch.no_grad():
            rollout_samples = self.rollout_samples(idx, prompt, ref_image_path, seed, n_samples=group_size)
        if self.p_state.cp_rank == 0:
            video_paths = [sample['video_path'] for sample in rollout_samples]
            score_dict, _ = self.reward_fn(video_paths, [prompt] * group_size)
            advantages = self.compute_advantages(group_size, score_dict, task_kwargs['reward_config'])
            self.logger.info(f"Reward score dict: {score_dict}, advantages: {advantages}")
            score_dict_list = [{k: torch.tensor(v[i]) for k, v in score_dict.items()} for i in range(group_size)]
        else:
            score_dict_list = [{}] * group_size
            advantages = torch.ones(group_size, device=self.device)
        dist.broadcast_object_list(score_dict_list, src=self.rank // self.p_state.cp_size * self.p_state.cp_size, group=self.p_state.cp_group)
        dist.broadcast(advantages, src=self.rank // self.p_state.cp_size * self.p_state.cp_size, group=self.p_state.cp_group)

        # prepare batches
        tk_duration, tk_height, tk_width = rollout_samples[0]['videos'].shape[-3:]
        tk_duration = tk_duration // args.patch_size
        tk_height = tk_height // args.patch_size
        tk_width = tk_width // args.patch_size
        rope_image_info = [
            [(None, (tk_duration, tk_height, tk_width), {'type':"gen_video"})]
        ]

        # get pairs of (win, lose) samples for DPO loss
        pairs = []
        pair = None
        max_score = -1
        dim1, dim2 = task_kwargs['dpo_reward_dims']
        for i in range(len(score_dict_list)):
            for j in range(len(score_dict_list)):
                if i==j:
                    continue
                if score_dict_list[i][dim1] >= score_dict_list[j][dim1]:
                    if pair is None or score_dict_list[i][dim2] - score_dict_list[j][dim2] > max_score:
                        max_score = score_dict_list[i][dim2] - score_dict_list[j][dim2]
                        reward_weight = 1.0
                        pair = (i, j, reward_weight)
        pairs = [pair]*task_kwargs.get('pair_life', 5)

        for pair in pairs:
            i, j, reward_weight = pair
            micro_batch = copy.deepcopy(batch)
            micro_batch.update({
                "win_latents": rollout_samples[i]['videos'].unsqueeze(0), # add batch dimension
                "lose_latents": rollout_samples[j]['videos'].unsqueeze(0), # add batch dimension
                "rope_media_info": rope_image_info,
            })
            micro_batches.append(micro_batch)
        return micro_batches


    def prepare_model_inputs(self, batch, device):
        return Leo2DPOTrainer.prepare_model_dpo_inputs(self, batch, device)

    def train_step(self, batch):
        return Leo2DPOTrainer.train_step(self, batch)