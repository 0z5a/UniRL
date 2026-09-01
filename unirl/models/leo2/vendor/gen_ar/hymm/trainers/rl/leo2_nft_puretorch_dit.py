import base64
import copy
import os
from time import time
import io
import json

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate

from hymm.data_kits.av_loader import MultimodalAVIndexDataset
from hymm.data_kits.utils.data_container import MultimodalDataContainer
from hymm.data_kits.utils.video_utils import VideoProcessor
from hymm.data_kits.utils.audio_utils import AudioProcessor


from hymm.models.reward_models import video_rewards
from hymm.core.global_vars import get_denoiser, get_mm_state, get_combined_iterator
from hymm.core.data_provider import DatasetsProvider
from hymm.core.data_provider_dit import prepare_model_t2v_inputs
from hymm.samplers.leo2_sampler import Leo2Sampler

from hymm.trainers.rl.leo2_dpo_puretorch_dit import Leo2DPOTrainer, CSVIndexManager, update_teacher_model

from processors.video_kits import save_video, save_video_audio

class VideoPromptDataset(MultimodalAVIndexDataset):

    def collate_fn(self, batch):
        result = {}
        for key in batch[0].keys():
            if key in ("rope_media_info", "dummy_type_dict"):
                result[key] = [item[key] for item in batch]
            else:
                result[key] = default_collate([item[key] for item in batch])
        return result


    def setup_index_manager(self, batch_size):
        required_cols = ['task_type', 'ref_image_path', 'prompt', 'seed']
        if self.task_kwargs["nft_mode"] == "mix_win":
            required_cols.extend(['win_video_path', 'win_vae_cache_path'])
        self.index_manager = CSVIndexManager(required_cols, self.index_kwargs, self.logger)
        self.video_processor = VideoProcessor(self.args)
        self.audio_processor = AudioProcessor(self.args)

    def get_batch(self, idx):
        args = self.args
        item = self.index_manager.datas[idx]
        task_type = item['task_type']

        ref_image_path = str(item['ref_image_path'])
        if task_type == "i2v" and not os.path.exists(ref_image_path):
            self.logger.warning(f"Reference image path {ref_image_path} does not exist for index {idx}.")
            task_type = "t2v"

        # Get prompts
        prompt = item['prompt']
        if args.prompt_prepend_content:
            prompt = args.prompt_prepend_content + prompt
        if args.prompt_append_content:
            prompt = prompt + args.prompt_append_content

        # Generate dummpy data
        if self.task_kwargs["nft_mode"] == "mix_win":
            latents_np = np.load(item['win_vae_cache_path']).squeeze(0)
            latents = torch.from_numpy(latents_np).to(torch.float16)
            dummy_video = self.vae_process_video(latents_np)
        else:
            video_size = self.task_kwargs['image_size']
            num_frames = self.task_kwargs['num_frames']
            info = self.video_processor.build_gen_video_info(video_size, num_frames)
            video_latent = np.zeros([1, info.token_duration, info.token_height, info.token_width])
            origin_size = [info.video_width, info.video_height, info.video_duration]
            dummy_video = self.as_video_tensor(video_latent, video_type=self.video_vae_info.video_type, origin_size=origin_size)

        if "vae_audio" in self.task_kwargs['modality']:
            waveform = torch.zeros(1, 16000)
            sr = 16000
            dummy_audio = self.vae_process_audio(waveform, sr, data_format="single_slice_file")
        else:
            dummy_audio = None
        data = MultimodalDataContainer(
            prompt=prompt,
            videos=[dummy_video],
            video_last_frames=None,
            audios=[dummy_audio] if dummy_audio is not None else None,
            success=True,
            index=idx,
            messages=[{"type": "gen_image", "text": prompt}]
        )
        sections = self.build_t2v_template(data)

        dummy_type_dict = {}
        if self.args.audio_branch_model_name is not None and \
            "vae_audio" not in self.task_kwargs['modality']:
            dummy_type_dict["audio"] = 1    # audio branch dummy token
        dummy_number = sum(dummy_type_dict.values())

        max_token_length = self.seq_length - dummy_number \
            if self.sequence_pack \
            else self.max_token_length - dummy_number

        output = self.tokenizer.encode_general(
            sections=sections,
            max_token_length=max_token_length,
            add_eos=False,
            drop_last=self.drop_last,
            add_pad=False if self.sequence_pack else 'auto',
            add_bos=self.default_conv.add_bos if hasattr(self, 'default_conv') else True,
            und_token_type=self.und_token_type,
            gen_token_type=self.gen_token_type,
            disable_ignore=True,
        )
        output.text_mask = output.text_mask.to(torch.long)
        rope_media_info, _ = self.get_rope_media_info(sections, output, data)

        result = {
            "dataset_tag": self.dataset_tag,
            "n_samples": 1,
            "index": idx,
            "tokens": output.tokens.clone(),
            "text_mask": output.text_mask.clone(),
            "rope_media_info": rope_media_info,
            "task_type": task_type,
            "text": prompt,
            "ref_image_path": item['ref_image_path'],
            "seed": item['seed'],
            "dummy_type_dict": dummy_type_dict,
        }
        if self.task_kwargs["nft_mode"] == "mix_win":
            result["videos"] = latents
            result["video_path"] = item['win_video_path']
        return result
        
    def __getitem__(self, idx):
        try_times = 10000
        for _ in range(try_times):
            try:
                return self.get_batch(idx)
            except Exception as e:
                self.logger.warning(f"Error loading sample {idx}: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError(f'Dataset errors occur {try_times} in continue __getitem__')

def nft_loss(x0, xt, ut, t_expanded, forward_prediction, old_prediction, r, beta, train_beta):
    '''
    x0: clean latent
    xt: noisy latent
    ut: velocity target for flow matching
    t_expanded: time step expanded
    forward_prediction: current model's prediction of noise
    old_prediction: teacher model's prediction of noise
    r: advantage score (0-1) for weighting positive and negative loss
    beta: weighting factor for positive and negative loss
    train_beta: weighting factor for KL divergence loss
    '''
    loss_terms = {}
    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()
    positive_prediction = beta * forward_prediction + (1 - beta) * old_prediction.detach()
    implicit_negative_prediction = (
        1.0 + beta
    ) * old_prediction.detach() - beta * forward_prediction

    # adaptive weighting
    x0_prediction = xt - t_expanded * positive_prediction
    with torch.no_grad():
        weight_factor = (
            torch.abs(x0_prediction.double() - x0.double())
            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
            .clip(min=0.00001)
        )
    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
    with torch.no_grad():
        negative_weight_factor = (
            torch.abs(negative_x0_prediction.double() - x0.double())
            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
            .clip(min=0.00001)
        )
    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
        dim=tuple(range(1, x0.ndim))
    )

    ori_policy_loss = r * positive_loss / beta + (1.0 - r) * negative_loss / beta
    policy_loss = ori_policy_loss.mean()

    loss = policy_loss
    loss_terms["policy_loss"] = policy_loss.detach()
    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

    kl_div_loss = ((forward_prediction - old_prediction) ** 2).mean(
        dim=tuple(range(1, x0.ndim))
    )

    loss += train_beta * torch.mean(kl_div_loss)
    kl_div_loss = torch.mean(kl_div_loss)
    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
    
    return loss, loss_terms
                     
class Leo2NFTTrainer(Leo2DPOTrainer):
    def __init__(self, args):
        super().__init__(args)
        run_task_kwargs = self.args.t2vi2v_task_kwargs

        self.reward_fn = video_rewards.create_reward_fn_from_config(
            run_task_kwargs['reward_config'], self.device, self.logger)

        dummy_trainer = copy.copy(self)
        dummy_trainer.model = self.teacher_model
        dummy_trainer.model_enginer = self.teacher_model_engine
        sampler = Leo2Sampler(None, self.rank, self.world_size, trainer=dummy_trainer)        
        #overwrite generation config with task specific kwargs if specified
        for key, value in run_task_kwargs.items():
            if hasattr(sampler.model.generation_config, key):
                setattr(sampler.model.generation_config, key, value)
        self.rollout_engine = sampler

    def build_dataloader(self):
        # Build dataloaders for multimodal tasks
        self.task_info_dict = {
            't2vi2v': dict(cur_task="videonft", cls=VideoPromptDataset),
        }
        DatasetsProvider(self.task_info_dict)(None)
        self.combined_iterator = get_combined_iterator()
        self.mm_state = get_mm_state()

    def rollout_samples(self, idx, prompt, ref_image_path, seed, n_samples, run_task_kwargs):
        args = self.args 
        bot_task = run_task_kwargs['bot_task']
        train_audio = "vae_audio" in run_task_kwargs['modality']

        videos = []
        update_steps = self.ss.update_steps
        save_dir = os.path.join(self.exp_dir, "rl_samples", "step{}".format(update_steps))
        teacher_update_decay = run_task_kwargs['teacher_update_decay']
        decay_warmup_steps = run_task_kwargs['decay_warmup_iters']
        if decay_warmup_steps > 0 and update_steps < decay_warmup_steps:
            teacher_update_decay = teacher_update_decay * update_steps / decay_warmup_steps
        os.makedirs(save_dir, exist_ok=True)
        for i in range(n_samples):
            # update_model_per_sample, update_model_per_group
            if run_task_kwargs.get("update_mode") == "update_model_per_sample":
                self.teacher_model_engine = update_teacher_model(self.model_engine, self.teacher_model_engine, decay=teacher_update_decay)
                self.logger.info(f"update teacher model with decay {teacher_update_decay} at iteration {i}/{n_samples}")
            else:
                if i==0:
                    self.teacher_model_engine = update_teacher_model(self.model_engine, self.teacher_model_engine, decay=teacher_update_decay)
                    self.logger.info(f"update teacher model with decay {teacher_update_decay} at iteration {i}/{n_samples}")
            self.logger.info(f"Rollout batch iteration {i+1}/{n_samples}, prompt: {prompt}, ref_image_path: {ref_image_path}, seed: {seed+i}")
            message_list = []
            img_str = None
            if ref_image_path is not None and type(ref_image_path) == str and os.path.exists(ref_image_path):
                message_list += [{
                        "role": "user",
                        "content": [{
                            "type": "image",
                            "image": ref_image_path,
                        }]
                    }]
                image_processor = self.rollout_engine.model.image_processor
                cond_image = image_processor.build_cond_images(message_list=message_list)[0]
                cond_image = image_processor.tensor_to_pil_image(cond_image)
                binary_image = io.BytesIO()
                cond_image.save(binary_image, format='JPEG')
                img_str = base64.b64encode(binary_image.getvalue()).decode("utf-8")


            message_list += [{"role": "user", "content": prompt}]
            outputs, latent_outputs = self.rollout_engine.model.generate_video(
                message_list=message_list, seed=seed+i, 
                video_size=run_task_kwargs["image_size"], 
                num_frames=run_task_kwargs["num_frames"], 
                video_fps=run_task_kwargs["video_fps"], 
                ref_mode=run_task_kwargs["ref_mode"], 
                output_type=dict(visual="np", audio="np"), 
                bot_task=bot_task,
                return_latents=True, verbose=1 if self.rank == 0 else 0
            )
            video_id = f"{idx}_{i}"
            if self.p_state.cp_rank == 0:
                save_path = os.path.join(save_dir, f"{video_id}.mp4")
                if outputs.audios is not None:
                    save_video_audio(outputs.videos[0], outputs.audios[0], save_path=save_path, 
                        fps=run_task_kwargs['video_fps'], sample_rate=args.audio_sample_rate)
                else:
                    save_video(outputs.videos[0], save_path=save_path, fps=run_task_kwargs['video_fps'])
                self.logger.info(f"Saved rollout video to {save_path}")
            else:
                save_path = None
            video_data = {
                "video_id": video_id,
                "video_path": save_path,
                "image_path": img_str,
                "videos": latent_outputs.videos[0].cpu(),
            }
            if train_audio and latent_outputs.audios is not None:
                video_data["audios"] = latent_outputs.audios[0].cpu()   
            videos.append(video_data)
        return videos

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

        # rollout samples
        group_size = int(task_kwargs['group_size'])
        with torch.no_grad():
            rollout_size = group_size - 1 if task_kwargs["nft_mode"] == "mix_win" else group_size
            rollout_samples = self.rollout_samples(idx, prompt, ref_image_path, seed, rollout_size, task_kwargs)
        # add extra high quality data
        if task_kwargs["nft_mode"] == "mix_win":
            video_data = {
                "video_id": f"{idx}_{rollout_size}",
                "video_path": batch["video_path"][0],
                "image_path": None,
                "videos": batch["videos"][0].cpu(),
            }
            self.logger.info(f"Adding extra high quality data for idx {idx} with video_path {batch['video_path'][0]}, shape {batch['videos'][0].shape}")
            rollout_samples.append(video_data)

        if self.p_state.cp_rank == 0:
            video_paths = [sample['video_path'] for sample in rollout_samples]
            meta_dict_list = [{"image_path": sample['image_path']} for sample in rollout_samples]
            score_dict, _ = self.reward_fn(video_paths, [prompt] * group_size, metadata=meta_dict_list)
            advantages = self.compute_advantages(group_size, score_dict, task_kwargs['reward_config'])
            self.logger.info(f"Reward score dict: {score_dict}, advantages: {advantages}")
            score_dict_list = [{k: torch.tensor(v[i]) for k, v in score_dict.items()} for i in range(group_size)]
            ## dump score_dict
            dump_path = video_paths[0].replace('.mp4', '.json')
            with open(dump_path, 'w') as f:
                json.dump({
                    'prompt': prompt,
                    'video_paths': video_paths,
                    'score_dict': score_dict,
                    'advantages': advantages.tolist()
                },f, ensure_ascii=False, indent=4)
        else:
            score_dict_list = [{}] * group_size
            advantages = torch.ones(group_size, device=self.device)
        dist.broadcast_object_list(score_dict_list, src=self.rank // self.p_state.cp_size * self.p_state.cp_size, group=self.p_state.cp_group)
        dist.broadcast(advantages, src=self.rank // self.p_state.cp_size * self.p_state.cp_size, group=self.p_state.cp_group)

        # prepare batches
        for i in range(group_size):
            tk_duration, tk_height, tk_width = rollout_samples[i]['videos'].shape[-3:]
            tk_duration = tk_duration // args.patch_size
            tk_height = tk_height // args.patch_size
            tk_width = tk_width // args.patch_size
            rope_media_info = batch['rope_media_info'][0]
            real_rope_media_info = [
                [
                (rope_media_info[0][0], (tk_duration, tk_height, tk_width), 
                        {'type':"gen_video", 'with_audio': rollout_samples[0].get('audios') is not None}),
                ]
            ]
            micro_batch = copy.deepcopy(batch)
            micro_batch.update({
                "idx": rollout_samples[i]['video_id'],
                "videos": rollout_samples[i]['videos'].unsqueeze(0), # add batch dimension
                "advantage": advantages[i],
                "rope_media_info": real_rope_media_info,
                "score_dict": score_dict_list[i],
            })
            micro_batches.append(micro_batch)
        return micro_batches


    def prepare_model_inputs(self, batch, device):
        with torch.autocast(device_type="cuda", enabled=False):
            model_input_kwargs, bsz, seqlen = prepare_model_t2v_inputs(batch, device, latent_channel_extend_type=batch['task_type'][0])
        return model_input_kwargs, batch['videos'], batch['advantage'], batch['score_dict'], bsz, seqlen


    def train_step(self, batch):
        args = self.args
        dataset_tag = batch["dataset_tag"][0]
        task_kwargs = getattr(args, f"{dataset_tag}_task_kwargs")
        vae = self.vae
        denoiser = get_denoiser()
        device = torch.device("cuda", args.local_rank)

        model_input_kwargs, clean_latent, advantage, score_dict, bsz, seqlen = self.prepare_model_inputs(batch, device)

        #calcuate loss
        with torch.no_grad():
            teacher_model_output = self.teacher_model_engine(**model_input_kwargs).diffusion_prediction
        x0 = clean_latent.to(self.device) * vae.config['scaling_factor']
        loss_fn = nft_loss

        def loss_closure(model_output, model_input_kwargs):
            current_model_output = model_output.diffusion_prediction
            channel_dim = current_model_output.shape[1]
            xt = model_input_kwargs['kwargs']['latents'][:, :channel_dim]
            ut = model_input_kwargs['kwargs']['ut']
            t_expanded = denoiser.get_scheduler_t(model_input_kwargs['kwargs']['timesteps'])
            loss, extra_info = loss_fn(
                x0=x0,
                xt=xt,
                ut=ut,
                t_expanded=t_expanded,
                forward_prediction=current_model_output,
                old_prediction=teacher_model_output,
                r=advantage,
                beta=task_kwargs['beta'],
                train_beta=task_kwargs['train_beta'],
            )
            extra_info['loss'] = loss
            return loss, extra_info

        self.model_engine.register_loss_closure(loss_closure)
        loss = self.model_engine(**model_input_kwargs)
        if torch.isnan(loss).any():
            self.nan_grad_count += 1
            self.logger.warning(f"NaN loss encountered in rank {self.rank}, total NaN count: {self.nan_grad_count}")

        loss_dict = self.model_engine.get_cached_result("loss_dict")
        loss_dict['loss'] = loss
        for key in score_dict.keys():
            loss_dict[key] = score_dict[key]
        for key in loss_dict.keys():
            if key not in self.loss_names:
                self.loss_names.append(key)

        consumed_metrics = {
            batch["dataset_tag"][0]: {
                "samples": batch["n_samples"].sum().item(),
                "tokens": bsz * seqlen
            }
        }

        return loss_dict, consumed_metrics
