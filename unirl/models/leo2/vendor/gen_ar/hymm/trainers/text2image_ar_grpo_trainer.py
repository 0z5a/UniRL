import os
import time
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np

from .helpers import dynamic_values_wrapper
from ..models import load_vae, build_model
from ..utils.file_utils import safe_dir
from ..data_kits.janus_text_image_loader import JanusTextImageArrowStream
from .text2image_ar_trainer import Text2ImageARTrainer
from ..utils.helpers import as_tuple
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE, move_model_params_and_grads_to
from ..samplers.janus_cot_ar_sampler import JanusCoTARSampler
from ..samplers.logits_processor import get_logits_processors
from ..samplers.image_processor_vlm import VLMImageProcessor
from ..models.reward_models.objs_counting_groundingdino import ObjectsCountingGroundingDino
from ..utils.deepspeed_utils import unwrap_model_for_generation_deepspeed
from ..data_kits.samplers import RepeatRandomDistributedSampler
from ..utils.torch_distributions import gather_tensor


class Text2ImageARGRPOTrainer(Text2ImageARTrainer):
    def __init__(self, args):
        self.num_generations = args.get("num_generations", 8)
        # TODO (yutaocui): Support num_grpo_iterations > 1, which is 𝜇 in the GRPO paper
        self.num_grpo_iterations = args.get("num_grpo_iterations", 1)
        assert (
            self.num_grpo_iterations == 1
        ), f"Invalid value of num_grpo_iterations ({self.num_grpo_iterations}), which should be 1."

        super().__init__(args)

        self.beta = args.get("kl_beta", 0.01)
        self.enable_think_mode = self.args.get("enable_think_mode", False)
        self.extra_und_penalty = self.args.get("extra_und_penalty", False)

        # Used to largely save memory
        self.model_cpu_offload = self.args.get("model_cpu_offload", False)
        self.reference_model_on_cpu = False

        # get special_token_id
        self.pad_token_id = self.dataset.tokenizer.pad_token_id
        self.cfg_token_id = self.dataset.tokenizer.cfg_token_id
        self.boi_token_id = self.dataset.tokenizer.boi_token_id
        self.eoi_token_id = self.dataset.tokenizer.eoi_token_id
        self.img_token_id = self.dataset.tokenizer.img_token_id
        self.gen_boi_token_id = self.dataset.tokenizer.gen_boi_token_id
        self.gen_eoi_token_id = self.dataset.tokenizer.gen_eoi_token_id
        self.gen_img_token_id = self.dataset.tokenizer.gen_img_token_id

    def create_reference_model(self):
        """
        Creates a static reference model. Note that model will be in `.eval()` mode.
        """
        self.logger.info("Building reference model...")
        factor_kwargs = {"device": self.device, "dtype": PRECISION_TO_TYPE[self.args.precision]}
        ref_model, _ = build_model(self.args, logger=self.logger, **factor_kwargs)
        ref_model.requires_grad = False
        return ref_model.eval()

    def build_extra_model(self):
        args = self.args
        # Build VAE
        self.vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=self.device,
            logger=self.logger,
        )
        self.image_token_offset = args.image_token_offset
        self.logger.info(f"Image token offset: {self.image_token_offset}")

        # Build processors
        self.answer_mode_image_logits_processor = None
        if args.get("answer_mode_image_logits_processors_cfg", None) is not None:
            answer_mode_image_logits_processors = get_logits_processors(args.answer_mode_image_logits_processors_cfg)
            self.answer_mode_image_logits_processor = answer_mode_image_logits_processors
        image_logits_processors = get_logits_processors(args.image_logits_processors_cfg)
        self.image_logits_processor = image_logits_processors
        text_logits_processors = get_logits_processors(args.text_logits_processors_cfg)
        self.text_logits_processor = text_logits_processors

        image_preprocessor = VLMImageProcessor.from_pretrained(args.image_preprocessor_pretrained_path)
        self.image_preprocessor = image_preprocessor

        # Build sampler
        self.sampler = self.get_sampler()

        # Build reference model
        beta = args.get("kl_beta", 0.01)
        if beta > 0:
            self.reference_model = self.create_reference_model()

        # Build reward model
        self.obj_count_reward_model = ObjectsCountingGroundingDino(box_thre=0.5, device=self.device)

    def get_sampler(self):
        train_format_prompt_type = self.args.get("train_format_prompt_type", "default")
        model_dict = dict(
            model_settings=self.model_settings,
            vae=self.vae,
            tokenizer=self.dataset.tokenizer,
            image_logits_processor=self.image_logits_processor,
            text_logits_processor=self.text_logits_processor,
            image_preprocessor=self.image_preprocessor,
            answer_mode_image_logits_processor=self.answer_mode_image_logits_processor,
        )

        sampler = JanusCoTARSampler(
            self.args,
            model_dict=model_dict,
            format_prompt_type=train_format_prompt_type,
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            logger=self.logger,
        )
        return sampler

    def _get_per_token_logps(self, model, model_intput_kwargs):
        """Get the per-token log probabilities of the completions for the current model and the reference model."""
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            output = model(**model_intput_kwargs)
        image_logits = output["image_logits"][:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        text_logits = output["text_logits"][:, :-1, :]
        input_ids = model_intput_kwargs["idx"][:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        text_mask = input_ids < self.image_token_offset

        text_log_probs = text_logits.log_softmax(dim=-1)  # (B, L-1, V_t)
        text_token_log_probs = torch.gather(
            text_log_probs,
            dim=2,
            index=input_ids.clamp(max=self.image_token_offset - 1).unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1)

        adjusted_image_indices = (input_ids - self.args.image_token_offset).clamp(min=0)
        image_log_probs = image_logits.log_softmax(dim=-1)  # (B, L-1, V_i)
        image_token_log_probs = torch.gather(
            image_log_probs,
            dim=2,
            index=adjusted_image_indices.unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1)

        per_token_logps = torch.where(text_mask, text_token_log_probs, image_token_log_probs)
        return per_token_logps

    def get_loss_mask(self, full_seq_token_tensor):
        no_loss_tokens = torch.tensor(
            [
                self.dataset.tokenizer.bos_token,
                self.boi_token_id,
                self.eoi_token_id,
                self.img_token_id,
                self.gen_eoi_token_id,
                self.pad_token_id,
                self.cfg_token_id,
            ]
        ).to(self.device)
        no_loss_mask = torch.isin(full_seq_token_tensor, no_loss_tokens)
        # TODO: Now only support `batch_size` equals to 1
        # if not self.enable_think_mode:
        # Find out the position of the first <gen_boi>
        gen_boi_positions = full_seq_token_tensor == self.gen_boi_token_id
        if gen_boi_positions.any():
            first_gen_boi_idx = torch.nonzero(gen_boi_positions, as_tuple=True)[1][0]
            first_gen_boi_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.bool)
            # Mask all before (including) the first <gen_boi>.
            first_gen_boi_mask[0, : (first_gen_boi_idx + 1)] = True
        else:
            first_gen_boi_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.bool)
        no_loss_mask = no_loss_mask | first_gen_boi_mask
        loss_mask = (~no_loss_mask).float()

        return loss_mask

    def prepare_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(batch, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))

        # ============================ 1. Generate samples using different seed =============================
        # TODO (yutaocui): Now only support `batch_size` equals to 1
        input_prompts = batch["text"]
        input_prompts = [prompt.split("<image_placeholder>")[0] for prompt in input_prompts]

        self.model_engine.module.eval()
        with torch.inference_mode():
            with unwrap_model_for_generation_deepspeed(self.model_engine) as unwrapped_model:
                self.sampler.model_dict["model"] = unwrapped_model
                outputs = self.sampler.predict(
                    input_prompts,
                    size=self.args.sample_image_size,
                    seed=None,  # set `seed` to None, so as to generate different samples in a batch
                )
        self.model_engine.module.train()

        last_answer = [outputs["last_answer"]]  # The last generated answer, e.g., "<answer> ... </answer>"
        last_pil_images = outputs["pil_images"]  # list, len = batch_size
        # completion_tokens: [B, L], includes the input tokens and the generated tokens
        completion_tokens = outputs["completion_ids"]
        und_images_tensor = outputs["und_images_tensor"]  # [1, N, C, H, W]
        eff_images_num = [und_images_tensor.shape[1]]
        # TODO: The max number of generated images is fixed to 3.
        if und_images_tensor.shape[1] < 3:
            pad_tensor = torch.zeros(1, 3 - und_images_tensor.shape[1], *und_images_tensor.shape[2:]).to(device)
            und_images_tensor = torch.cat([und_images_tensor, pad_tensor], dim=1)

        # Pad to the max number of tokens
        completion_tokens = completion_tokens[:, : self.args.max_generated_tokens]
        pad_num = self.args.max_generated_tokens - completion_tokens.shape[1]
        if pad_num > 0:
            pad_ids = torch.tensor([self.pad_token_id] * pad_num).unsqueeze(0).to(device)
            completion_tokens = torch.cat([completion_tokens, pad_ids], dim=1)  # [B, max_len_tokens]

        # Get image loss mask and text loss mask
        loss_mask = self.get_loss_mask(completion_tokens)[:, 1:]

        # ===================================== 2. Pack model kwargs ========================================
        model_input_kwargs = dict(
            idx=completion_tokens.detach(),  # [b, l]
            imgs_input=und_images_tensor.detach(),  # [b, 3, c, h, w]
            image_token_id=self.img_token_id,
            eff_images_num=eff_images_num,  # list(*), len = b
            return_loss=False,
        )

        # ======================== 3. Get ref_token logits with the reference model =========================
        ref_per_token_logps = None
        if self.beta > 0:
            if self.model_cpu_offload and self.reference_model_on_cpu:
                self.reference_model_on_cpu = False
                self.reference_model = move_model_params_and_grads_to(self.reference_model, self.device)
            with torch.inference_mode():
                ref_per_token_logps = self._get_per_token_logps(self.reference_model, model_input_kwargs)
            if self.model_cpu_offload:
                self.reference_model_on_cpu = True
                self.reference_model = move_model_params_and_grads_to(self.reference_model, "cpu")
            # Note: Tokens with IDs above image_token_offset are treated as visual embeddings
            # and calculated using image_logits, while others use text_logits. Special tokens
            # (e.g., <pad>, <eos>) are explicitly masked out during loss computation to prevent
            # gradient contamination.
            ref_per_token_logps = ref_per_token_logps * loss_mask

        batch_size = completion_tokens.shape[0]
        n_tokens = completion_tokens.shape[1]
        obj = batch["object"]

        return (
            model_input_kwargs,
            loss_mask,
            last_answer,
            last_pil_images,
            ref_per_token_logps,
            obj,
            input_prompts,
            batch_size,
            n_tokens,
        )

    def train_step(self, batch):
        start1 = time.time()
        (
            model_input_kwargs,
            loss_mask,
            last_answer,
            last_pil_images,
            ref_per_token_logps,
            obj,
            input_prompts,
            cur_batch_size,
            n_tokens,
        ) = self.prepare_model_inputs(batch, self.device)
        torch.cuda.synchronize()
        duration1 = time.time() - start1

        start2 = time.time()
        per_token_logps = self._get_per_token_logps(self.model_engine, model_input_kwargs)
        per_token_logps = per_token_logps * loss_mask

        # Compute the KL divergence between the model and the reference model
        per_token_kl = 0.0
        if self.beta > 0:
            ref_per_token_logps = ref_per_token_logps.detach()
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Get advantages based on the rewards
        rewards = self.obj_count_reward_model(
            last_pil_images,
            obj,
            last_answer,
            input_prompts,
            self.extra_und_penalty,
            self.enable_think_mode,
        )
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device).detach()
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        gathered_rewards = gather_tensor(rewards).view(-1)  # [world_size, batch_size]
        self.logger.info(f"gathered_rewards: {gathered_rewards}")

        # Compute grouped-wise rewards
        mean_grouped_rewards = gathered_rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = gathered_rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (gathered_rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            dist.get_rank() * len(input_prompts),
            (dist.get_rank() + 1) * len(input_prompts),
        )
        advantages = advantages[process_slice]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.detach().unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)

        torch.cuda.synchronize()
        duration2 = time.time() - start2

        loss_dict = {"loss": loss, "advantage_dummyloss": advantages, "reward_dummyloss": rewards}
        self.logger.info(loss_dict)
        times = {
            "preprocess": duration1,
            "forward": duration2,
        }

        return loss_dict, cur_batch_size, n_tokens, times

    def build_dataloader(self):
        args = self.args
        self.dataset = JanusTextImageArrowStream(
            args=args,
            index_file=args.index_file,
            training_image_size=args.training_image_size,
            image_token_length=args.image_token_length,
            image_token_offset=args.get("image_token_offset", 0),
            use_pre_extracted_token=args.get("use_pre_extracted_token", False),
            text_token_length=args.text_token_length,
            uncond_p=args.uncond_p,
            tokenizer_name=args.tokenizer_name,
            multireso=args.multireso,
            add_iw_ih_token=args.add_iw_ih_token,
            index_kwargs=dict(
                batch_size=1 if args.mix_scale else self.micro_batch_size,
                world_size=1 if args.mix_scale else self.world_size,
                **args.index_kwargs,
            ),
            debug=False,
            logger=self.logger,
        )
        # Build sampler and data loader
        dataloader_kwargs = dict(**args.dataloader_params, worker_init_fn=set_worker_seed_builder(self.rank))
        
        if args.mix_scale:
            raise NotImplementedError
        else:
            if args.multireso:
                raise NotImplementedError
            else:
                self.data_sampler = RepeatRandomDistributedSampler(
                    self.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    seed=args.global_seed,
                    drop_last=True,
                    mini_repeat_count=self.num_generations,
                    repeat_count=self.num_grpo_iterations,
                    batch_size=self.micro_batch_size,
                )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.micro_batch_size,
                sampler=self.data_sampler,
                shuffle=False,
                drop_last=True,
                **dataloader_kwargs,
            )
