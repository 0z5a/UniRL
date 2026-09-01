import re
from typing import Dict, List

import PIL.Image
import torch
from einops import rearrange

from hymm.samplers.base_sampler import BaseSampler, setup_distributed_initialize, textimage2text_interactive
from hymm.samplers.logits_processor import get_logits_processors
from hymm.config import parse_eval_initial_args
from hymm.models import load_vae, TokenizerWrapper
from hymm.utils.file_utils import rank0_logger
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.samplers.image_processor_vlm import VLMImageProcessor, load_pil_images


class ImageUnderstandingARSampler(BaseSampler):
    def __init__(self, args, model_dict, ckpt_path=None, rank=0, world_size=1, device=0, logger=None):
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
        # whether to use separate wte and head for image generation tokens
        self.enable_sep_modal_adaptors = args.get("enable_sep_modal_adaptors", False)
        for logits_processor in args.logits_processors_cfg:
            for name, _ in logits_processor.items():
                if name == "CfgLogitsWarper":
                    self.cfg_enabled = True

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
        # Used for Janus models.
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
        logits_processors = get_logits_processors(args.logits_processors_cfg)
        model_dict["logits_processor"] = logits_processors

        # =========================== Build vlm_image_preprocessor ======================
        image_preprocessor = VLMImageProcessor.from_pretrained(args.image_preprocessor_pretrained_path)
        model_dict["image_preprocessor"] = image_preprocessor

        return model_dict
    
    def format_prompt(self, prompt, format_type="default"):
        if format_type == "janus-understanding":
            formatted_prompt = f"You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: {prompt}\n\nAssistant:"
        else:
            formatted_prompt = prompt
        return formatted_prompt
    
    def parse_original_prompt(self, text, image_token_tag='<image>'):
        """
        The original prompt should in format like:
            "<input_description>: text <image> <image>. <input_images>: the image path1; the image path2"
        """
        pattern = r'<input_description>:\s*(.*?)\s*<input_images>:\s*(.*)'
        match = re.search(pattern, text)
        
        if not match:
            return None, []
        
        prompt = match.group(1).strip()
        prompt = prompt.replace("<image>", image_token_tag)
        num_images = prompt.count(image_token_tag)
        
        images_path_str = match.group(2).strip()
        images_path = [path.strip() for path in images_path_str.split(';') if path.strip()]
        assert num_images == len(images_path)
        
        return prompt, images_path
    
    @torch.no_grad()
    def predict(self, prompt, **kwargs):
        tokenizer = self.model_dict["tokenizer"]
        img_token_tag = self.args.get("img_token_tag", None)
        eos_token_tag = self.args.get("eos_token_tag", None)

        # TODO: Support batch_size > 1
        prompt, imgs_path = self.parse_original_prompt(prompt, image_token_tag=img_token_tag)
        format_prompt_type = self.args.get("format_prompt_type", "default")
        prompt = self.format_prompt(prompt, format_type=format_prompt_type)
        condition_dict, batch_size = self.prepare_prompts(prompt, negative_prompt=None, **kwargs)
        prompt = condition_dict['prompt']
        self.logger.info(f"formatted prompt: {prompt}")

        pil_imgs = load_pil_images(imgs_path)
        imgs_input = self.model_dict["image_preprocessor"](pil_imgs, return_tensors="pt").pixel_values # [N, C, H, W]
        imgs_input = imgs_input.unsqueeze(0) # [B, N, C, H, W]

        seeds = self.prepare_seed(
            seed=kwargs.get('seed', None),
            batch_size=batch_size,
            num_sample_per_prompt=1,
        )
        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        image_seq_len = self.args.image_token_length
        imgs_token_length = [image_seq_len] * len(pil_imgs)

        # 'encode_ar_ti2t_infer' is used for janus-series infer
        tokenizer_infer_fn = self.args.get("tokenizer_infer_fn", "defualt")
        if tokenizer_infer_fn == "encode_ar_ti2t_infer":
            infer_fn = tokenizer.encode_ar_ti2t_infer
            infer_fn_kwargs_list = [dict(
                imgs_token_length=imgs_token_length,
                img_token_tag=img_token_tag,
            ) for _ in range(batch_size)]
        else:
            raise KeyError(f"The 'tokenizer_infer_fn' of {tokenizer_infer_fn} is not supported for multimodal uderstanding.")
        input_ids, _, batched_real_pos = tokenizer.batch_gen_infer(
            infer_fn=infer_fn,
            prompt_list=prompt,
            infer_fn_kwargs_list=infer_fn_kwargs_list,
            do_classifier_free_guidance=False,
        )
        input_ids = input_ids.to(self.device)
        batched_real_pos = batched_real_pos.to(self.device)

        model_intput_kwargs = {
            "imgs_input": imgs_input,
            "image_token_id": tokenizer.tokenizer.convert_tokens_to_ids(img_token_tag),
        }

        # TODO: Support kv cache infer.
        self.model_dict["model"].set_kv_cache(batch_size=input_ids.shape[0], device=self.device)
        token_list = []
        iteration = 0
        with torch.no_grad():
            while True:
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                    output = self.model_dict["model"](input_ids, **model_intput_kwargs)
                if not self.enable_sep_modal_adaptors:
                    logits = output["logits"]
                else:
                    logits = output["text_logits"]
                next_token = logits[0, -1].argmax(dim=-1).cpu().item()
                print(next_token)
                token_list.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(self.device)], dim=1)
                if next_token == tokenizer.eos_token:
                    break
                iteration += 1
                if iteration > 512:
                    break

        # must clear kv cache in interactive mode
        self.model_dict["model"].clear_kv_cache()
        answer = tokenizer.tokenizer.decode(token_list, skip_special_tokens=True)
        
        return {"samples": answer}


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)

    sampler = ImageUnderstandingARSampler.from_pretrained(
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
        textimage2text_interactive(args, sampler, logger)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
