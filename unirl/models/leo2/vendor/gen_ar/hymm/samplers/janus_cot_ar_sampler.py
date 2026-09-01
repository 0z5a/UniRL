import re
import os
import time
from typing import Dict, List

import numpy as np
import PIL.Image
import pandas as pd
import torch
from einops import rearrange

from hymm.samplers.base_sampler import (
    BaseSampler,
    setup_distributed_initialize,
    text2image_interactive,
    text2image_batch,
)
from hymm.samplers.logits_processor import get_logits_processors
from hymm.config import parse_eval_initial_args
from hymm.models import load_vae, TokenizerWrapper
from hymm.utils.file_utils import rank0_logger
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.samplers.image_processor_vlm import VLMImageProcessor, load_pil_images


THINK_SYSTEM_PROMPT = (
    "The user provides a prompt, and the assistant generates images and verifies whether the number of objects "
    "in the generated images is accurate. The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. The reasoning process and the final generated image are enclosed "
    "within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> the generated image </answer>\n\nUser: Generate an image of '{}'. Assistant: "
)


class JanusCoTARSampler(BaseSampler):
    def __init__(
        self, args, model_dict, format_prompt_type=None, ckpt_path=None, rank=0, world_size=1, device=0, logger=None
    ):
        super().__init__(
            args=args,
            model_dict=model_dict,
            ckpt_path=ckpt_path,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )
        self.vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
        self.image_token_offset = args.image_token_offset
        self.vae_downsample_factor = model_dict["vae"]._downsample_factor
        self.vae_trans_type = model_dict["vae"]._trans_type
        self.cfg_enabled = False
        self.format_prompt_type = format_prompt_type
        # whether to use separate wte and head for image generation tokens
        self.enable_sep_modal_adaptors = args.get("enable_sep_modal_adaptors", False)
        for logits_processor in args.image_logits_processors_cfg:
            for name, _ in logits_processor.items():
                if name == "CfgLogitsWarper":
                    self.cfg_enabled = True

        self.enable_think_mode = self.args.get("enable_think_mode", False)
        if self.enable_think_mode:
            self.args.format_prompt_type = "think"

    @staticmethod
    def build_extra_model(args, model_dict, factor_kwargs, logger=None):
        vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=factor_kwargs["device"],
            logger=logger,
        )
        model_dict["vae"] = vae

        # =========================== Build text tokenizer ========================
        patch_special_tokens_dict = {
            "img": args.get("img_token_tag", None),
            "pad": args.get("pad_token_tag", None),
            "cfg": args.get("cfg_token_tag", None),
            "boi": args.get("boi_token_tag", None),
            "eoi": args.get("eoi_token_tag", None),
        }
        tokenizer = TokenizerWrapper(args.tokenizer_name, logger, patch_special_tokens_dict)
        model_dict["tokenizer"] = tokenizer

        # =========================== Build logits_processor ======================
        if args.get("answer_mode_image_logits_processors_cfg", None) is not None:
            answer_mode_image_logits_processors = get_logits_processors(args.answer_mode_image_logits_processors_cfg)
            model_dict["answer_mode_image_logits_processor"] = answer_mode_image_logits_processors
        image_logits_processors = get_logits_processors(args.image_logits_processors_cfg)
        model_dict["image_logits_processor"] = image_logits_processors
        text_logits_processors = get_logits_processors(args.text_logits_processors_cfg)
        model_dict["text_logits_processor"] = text_logits_processors

        # =========================== Build vlm_image_preprocessor ======================
        image_preprocessor = VLMImageProcessor.from_pretrained(args.image_preprocessor_pretrained_path)
        model_dict["image_preprocessor"] = image_preprocessor

        return model_dict

    def format_prompt(self, prompt, format_type="default"):
        if format_type == "default":
            prompt = prompt.replace("a photo of", "")
            # prompt = prompt.replace(" a ", " one ").replace(" an ", " one ").strip()  # 数据集中描述"一个"都是one
            formatted_prompt = f"You are a helpful vision generation and understanding assistant. \n\nUser: Generate image of '{prompt}'\n\nAssistant: "
        elif format_type == "think":
            prompt = prompt.replace("a photo of", "")
            # prompt = prompt.replace(" a ", " one ").replace(" an ", " one ").strip()
            formatted_prompt = THINK_SYSTEM_PROMPT.format(prompt)
        else:
            formatted_prompt = prompt
        return formatted_prompt

    @torch.no_grad()
    def get_next_token(self, x, input_pos, generators, no_iw_ih_scatter, pred_mode, **model_intput_kwargs):
        """
        Generate the next token based on the current input and prediction mode.
        """
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_enabled):
            outs = self.model_dict["model"](
                x, input_pos=input_pos, no_iw_ih_scatter=no_iw_ih_scatter, **model_intput_kwargs
            )
        
        # Select logits based on prediction mode
        if not self.enable_sep_modal_adaptors:
            logits = outs["logits"]
        else:
            if "answer_image_gen" == pred_mode:
                logits = outs["image_logits"]
                logits = logits[:, -1, :]
                logits = self.model_dict["answer_mode_image_logits_processor"](x, logits)
            elif "image_gen" in pred_mode:
                logits = outs["image_logits"]
                logits = logits[:, -1, :]
                logits = self.model_dict["image_logits_processor"](x, logits)
            else:
                logits = outs["text_logits"]
                logits = logits[:, -1, :]
                logits = self.model_dict["text_logits_processor"](x, logits)

        if self.cfg_enabled:
            logits, _ = torch.chunk(logits, chunks=2, dim=0)

        if "image_gen" not in pred_mode and not self.args.text_do_sample:
            next_token = logits.argmax(dim=-1)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_list = []
            for batch_idx, generator in enumerate(generators):
                next_token_list.append(torch.multinomial(probs[batch_idx], num_samples=1, generator=generator))
            next_token = torch.cat(next_token_list, dim=0).unsqueeze(-1)

        if self.cfg_enabled:
            next_token = next_token.repeat((2, 1))
        return next_token

    def tensors2pil_images(self, images):
        """
        images can be:
        * torch.Tensor, shape (B, C, H, W)
        * numpy.ndarray, values in interval [0, 1], shape (H, W, C) or (B, H, W, C)
        * a list of PIL.Image.
        """
        if isinstance(images, torch.Tensor):
            # Tensor -> numpy.ndarray
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        if isinstance(images, np.ndarray):
            # numpy.ndarray -> PIL.Image
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            if images.shape[-1] == 1:
                # special case for grayscale (single channel) images
                images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in images]
            else:
                images = [PIL.Image.fromarray(image) for image in images]

        return images

    def decode_image_and_save(self, image_token_ids, latent_h, latent_w, prompt, gen_id=0, save_results=False):
        image_token_ids = rearrange(image_token_ids, "b (h w) -> b h w", h=latent_h, w=latent_w)
        with torch.autocast(device_type="cuda", dtype=self.vae_dtype, enabled=self.vae_dtype != torch.float32):
            results_img = self.model_dict["vae"].vq_decode(image_token_ids)
        if self.vae_trans_type == "-11":
            results_img = results_img.clamp(-1, 1) * 0.5 + 0.5
        elif self.vae_trans_type == "01":
            results_img = results_img.clamp(0, 1)

        pil_images = self.tensors2pil_images(results_img)
        if save_results:
            save_name = f"{prompt.replace(' ', '_')[:20]}_{gen_id}".replace(".", "")
            inter_save_dir = f"{self.args.sample_save_path}_inter_results"
            save_paths = [f"{inter_save_dir}/{save_name}_{int(time.time())}.png" for i in range(len(results_img))]
            self.save_batch_image(results_img, save_paths)
            self.logger.info(f"Save the generated image to: {save_paths}")

        return results_img, pil_images

    def save_answer(self, answer):
        save_answer_file = f"{self.args.sample_save_path}_inter_results/answer.csv"
        if os.path.exists(save_answer_file):
            existing_data = pd.read_csv(save_answer_file)
            row_id = len(existing_data) + 1
        else:
            row_id = 1
        new_data = pd.DataFrame({"row_id": [row_id], "answer": [answer]})
        new_data.to_csv(save_answer_file, mode="a", header=not os.path.exists(save_answer_file), index=False)

    @torch.no_grad()
    def predict(self, prompt, **kwargs):
        # TODO (yutaocui): Now only support: `batch_size` equals to 1
        tokenizer = self.model_dict["tokenizer"]
        think_token_id = tokenizer.get_special_token_id(special_token_tag="<think>")
        answer_token_id = tokenizer.get_special_token_id(special_token_tag="<answer>")

        prompt = prompt[0] if isinstance(prompt, list) else prompt
        ori_prompt = prompt
        format_prompt_type = (
            self.format_prompt_type if self.format_prompt_type is not None else self.args.format_prompt_type
        )
        prompt = self.format_prompt(ori_prompt, format_type=format_prompt_type)
        condition_dict, batch_size = self.prepare_prompts(prompt, negative_prompt=None, **kwargs)
        prompt = condition_dict["prompt"]
        self.logger.info(f"formatted prompt: {prompt}")

        seeds = self.prepare_seed(
            seed=kwargs.get("seed", None),
            batch_size=batch_size,
            num_sample_per_prompt=1,
        )
        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        image_size = self.parse_image_size(kwargs["size"])
        latent_h = int(image_size[0] / self.vae_downsample_factor[0])
        latent_w = int(image_size[1] / self.vae_downsample_factor[1])

        infer_fn = tokenizer.encode_ar_text_only_infer
        if self.enable_think_mode:
            # Note: extra_special_tokens is placed to be not affected by the `uncond_p`
            extra_special_tokens = [tokenizer.special_token_map["<think>"], tokenizer.gen_boi_token_id]
        else:
            extra_special_tokens = [tokenizer.gen_boi_token_id]
        infer_fn_kwargs_list = [{"extra_special_tokens": extra_special_tokens} for _ in range(batch_size)]

        input_ids, _, batched_real_pos = tokenizer.batch_gen_infer(
            infer_fn=infer_fn,
            prompt_list=prompt,
            infer_fn_kwargs_list=infer_fn_kwargs_list,
            do_classifier_free_guidance=self.args.guidance_scale > 1.0,
        )
        input_ids = input_ids.to(self.device)
        batched_real_pos = batched_real_pos.to(self.device)
        # the `completion_ids` contains the input_ids and all the generated ids, maybe used for GRPO training
        completion_ids = input_ids[:1].clone()

        input_pos = torch.arange(0, input_ids.shape[1], device=self.device, dtype=torch.long)
        current_pos = batched_real_pos

        model_intput_kwargs = {
            "image_token_id": tokenizer.img_token_id,
        }

        text_token_list = []
        last_text_token_list = []
        self.save_inter_results = self.args.get("save_inter_results", False)
        
        # Initialize model and variables
        self.model_dict["model"].set_kv_cache(batch_size=input_ids.shape[0], device=self.device)
        completion_und_images_tensor = torch.empty((1, 0, 3, self.args.training_image_size, self.args.training_image_size), device=self.device)
        # pred_mode("answer_image_gen", "answer", "think", "image_gen", "default") initialization
        pred_mode = "image_gen"
        image_token_ids = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=self.device)
        cur_gen_token_num = 0
        total_gen_imgs_num = 0

        with torch.no_grad():
            while True:
                # Generate the next token
                next_token = self.get_next_token(
                    input_ids,
                    input_pos,
                    generators=generators,
                    no_iw_ih_scatter=True,
                    pred_mode=pred_mode,
                    **model_intput_kwargs,
                )

                if "image_gen" in pred_mode:
                    image_token_ids = torch.cat([image_token_ids, next_token.clone().view(-1, 1)], dim=-1)
                    next_token = next_token + self.image_token_offset
                    cur_gen_token_num += 1
                else:
                    text_token_list.append(next_token[:1].cpu().item())
                    last_text_token_list.append(next_token[:1].cpu().item())
                    answer = tokenizer.tokenizer.decode(text_token_list, skip_special_tokens=False)
                    self.logger.info(f"inter-answer: {answer}")
                
                # Update input IDs and positions
                input_ids = next_token.view(input_ids.shape[0], 1)
                model_intput_kwargs = {
                    "image_token_id": tokenizer.img_token_id,
                }
                current_pos += 1
                input_pos = current_pos.clone()
                completion_ids = torch.cat([completion_ids, input_ids[:1]], dim=1)

                # Handle completed image generation
                if cur_gen_token_num == self.args.image_token_length:
                    total_gen_imgs_num += 1
                    results_img, pil_imgs = self.decode_image_and_save(
                        image_token_ids[:1],
                        latent_h=latent_h,
                        latent_w=latent_w,
                        prompt=ori_prompt,
                        gen_id=total_gen_imgs_num,
                        save_results=self.save_inter_results,
                    )

                    # Add tokens for image understanding or answer generation
                    auto_add_tokens = [tokenizer.gen_eoi_token_id]
                    current_pos += 1  # 1 for <gen_eoi>
                    if not (self.enable_think_mode and pred_mode == "answer_image_gen"):
                        auto_add_tokens.append(tokenizer.boi_token_id)
                        auto_add_tokens.extend([tokenizer.img_token_id] * self.args.image_token_length)
                        auto_add_tokens.append(tokenizer.eoi_token_id)
                        current_pos += 2 + self.args.image_token_length  # 2 for <boi> <eoi>

                        # Process image embeddings for understanding
                        imgs_input = self.model_dict["image_preprocessor"](pil_imgs, return_tensors="pt").pixel_values  # [N, C, H, W]
                        imgs_input = imgs_input.unsqueeze(0).to(self.device)  # [B, N, C, H, W]
                        completion_und_images_tensor = torch.cat([completion_und_images_tensor, imgs_input], dim=1)
                        und_images_tensor = imgs_input.repeat(input_ids.shape[0], 1, 1, 1, 1)
                        model_intput_kwargs.update({
                            "imgs_input": und_images_tensor,
                            "eff_images_num": [1] * input_ids.shape[0],
                        })

                        # Switch to the text prediction mode
                        pred_mode = "default"
                        last_text_token_list = []
                    else:
                        last_text_token_list.extend([tokenizer.gen_img_token_id] * self.args.image_token_length)
                        last_text_token_list.append(tokenizer.gen_eoi_token_id)
                        pred_mode = "answer"

                    # Reset image token IDs and update positions
                    image_token_ids = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=self.device)
                    cur_gen_token_num = 0
                    input_pos = torch.arange(
                        input_pos[0].cpu().item() - 1, current_pos[0].cpu().item(), device=self.device, dtype=torch.long
                    )
                    auto_add_tokens = torch.tensor([auto_add_tokens]).repeat(input_ids.shape[0], 1).to(self.device)
                    input_ids = torch.cat([input_ids, auto_add_tokens], dim=1)
                    completion_ids = torch.cat([completion_ids, auto_add_tokens[:1]], dim=1)  # bs=1

                    assert input_ids.shape[1] == input_pos.shape[0], (
                        f"The `input_ids` length ({input_ids.shape[1]}) does not match the `input_pos` length ({input_pos.shape[0]})."
                    )

                # Handle mode switching
                elif next_token[:1].cpu().item() == tokenizer.gen_boi_token_id:
                    pred_mode = "answer_image_gen" if pred_mode == "answer" else "image_gen"
                    image_token_ids = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=self.device)
                    cur_gen_token_num = 0

                # Handle eos token
                elif next_token[:1].cpu().item() == tokenizer.eos_token:
                    last_text_token_list = last_text_token_list[:-1]  # remove <eos>
                    break
                
                # Handle think and answer modes
                elif self.enable_think_mode:
                    if next_token[:1].cpu().item() == think_token_id:
                        pred_mode = "think"
                    elif next_token[:1].cpu().item() == answer_token_id:
                        pred_mode = "answer"
                        last_text_token_list = [answer_token_id]
                        think_last_results_img = results_img

                # Terminal signals
                if completion_ids.shape[-1] >= self.args.max_generated_tokens:
                    break

        self.logger.info(f"Total length of tokens is {completion_ids.shape[-1]}")

        # must clear kv cache in interactive mode
        self.model_dict["model"].clear_kv_cache()
        answer = tokenizer.tokenizer.decode(text_token_list, skip_special_tokens=False)
        last_answer = tokenizer.tokenizer.decode(
            last_text_token_list, skip_special_tokens=False if self.enable_think_mode else True
        )
        # self.logger.info(f"last answer: {last_answer}")
        self.logger.info(answer)
        if self.save_inter_results:
            self.save_answer(answer)

        return {
            "samples": results_img,  # [B, C, H, W]
            "pil_images": pil_imgs,
            "completion_ids": completion_ids,  # [B, L]
            "und_images_tensor": completion_und_images_tensor,  # [B, N, C, H, W]
            "last_answer": last_answer,
        }


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)

    sampler = JanusCoTARSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args

    # Start evaluation
    # TODO (yutaocui): support csv sampling
    if args.interactive:
        text2image_interactive(args, sampler, logger)
    elif args.csv:
        # add logits processor config into segments automatically, cfg is included
        segments = []
        logits_processor_list = args.get("image_logits_processors_cfg", [])
        for logits_processor in logits_processor_list:
            for name, kwargs in logits_processor.items():
                if name in ["CfgLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"]:
                    for k, v in kwargs.items():
                        seg = f"{k}{v}"
                        segments.append(seg)
        
        text2image_batch(
            args=args,
            datasource=args.csv,
            sampler=sampler,
            logger=logger,
            segments=segments,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
