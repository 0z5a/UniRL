import time

import torch
from pathlib import Path
from index_kits.resolution import ResolutionGroup

from hymm.ar import load_pipeline
from hymm.config import parse_eval_initial_args
from hymm.data_kits.arrow_dataset import ArrowDataset
from hymm.models import load_vae, TokenizerWrapper, Arrangement
from hymm.samplers.base_sampler import BaseSampler
from hymm.samplers.base_sampler import (
    setup_distributed_initialize,
    label2image_interactive,
    text2image_interactive,
    label2image_batch,
    text2image_batch,
)
from hymm.samplers.logits_processor import get_logits_processors
from hymm.utils.file_utils import rank0_logger, safe_json
from hymm.utils.helpers import to_2tuple, default
from hymm.utils.torch_utils import PRECISION_TO_TYPE, set_manual_seed


class Text2ImageMLMSampler(BaseSampler):
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

        if model_dict.get('arrange', None) is None:
            pipeline_name = "maskgit"
        else:
            pipeline_name = default(args.pipeline, "mlm")
        self.pipeline = load_pipeline(args, name=pipeline_name, rank=rank, device=device, **model_dict)

    @classmethod
    def build_extra_model(cls, args, model_dict, factor_kwargs, logger=None):
        # The phrase `vae` is used to denote the VQ-VAE/VQ-GAN series models. It is not a VAE in the traditional sense.
        vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=factor_kwargs["device"],
            logger=logger,
        )
        model_dict["vae"] = vae

        if args.get('label', None) is not None:
            return model_dict

        tokenizer = TokenizerWrapper(args.tokenizer_name, logger)
        model_dict['tokenizer'] = tokenizer

        # ====================== Token Specification ======================
        image_size = to_2tuple(args.image_size)
        text_vocab_size = model_dict["model_settings"].padded_vocab_size
        image_id_range = (-1, -1) if args.pipeline == "mar" else (text_vocab_size, text_vocab_size + vae.codebook_size)
        arrange = Arrangement(
            text_id_range=(0, text_vocab_size),
            image_id_range=image_id_range,
            pad_id=tokenizer.special_token_map["<pad>"],
            img_id=tokenizer.special_token_map["<img>"],
            mask_id=tokenizer.special_token_map["<mask>"],
            uncond_id=tokenizer.special_token_map["<cfg>"],
            boi_id=tokenizer.special_token_map["<boi>"],
            eoi_id=tokenizer.special_token_map["<eoi>"],
            s_text_range=(None, args.text_token_length - 3),
            s_text_maxlen=args.text_token_length,
            s_image_range=(args.text_token_length - 2, None),
            s_image_maxlen=(image_size[0] * image_size[1]) // vae.downsample_factor ** 2,
            sequence=["<pad>", "<bos>", "<text>", "<boi>", "<image>", "<eoi>", "<eos>"],
            ignore_id=args.ignore_index,
        )
        model_dict['arrange'] = arrange

        # =========================== Build logits_processor ======================
        if args.pipeline == 'mlm':
            logits_processors = get_logits_processors(args.get('logits_processors_cfg', []))
            model_dict["logits_processor"] = logits_processors

        return model_dict

    def get_sample_dir_suffix(self, task=None, testset=None, image_size=None, load_key=None, segments=None, rerank=None):
        args = self.args
        pp = args.get('pipeline_kwargs', {})
        segments = [
            f"{args.schedule_type}" if args.schedule_type != 'shift' else f"shift{args.mask_ratio_shift}",
            f"step{args.infer_steps}",
            f"cfg{args.guidance_scale}",
            f"t{default(args.temperature, pp.get('temperature', 4.5))}",
        ] + default(segments, [])
        return super().get_sample_dir_suffix(
            task=task,
            testset=testset,
            image_size=image_size,
            load_key=load_key,
            segments=segments,
            rerank=rerank,
        )

    def get_infer_kwargs(self, safe=False):
        args = self.args
        pp = args.get('pipeline_kwargs', {})
        logits_processor_kwargs = {}
        if args.get('logits_processors_cfg', []):
            logits_processor_kwargs['top_k'] = args.top_k
            logits_processor_kwargs['top_p'] = args.top_p
        kwargs = dict(
            infer_steps=args.infer_steps,
            pipeline_kwargs={
                'guidance_dynamic': default(args.guidance_dynamic, pp.get('guidance_dynamic', 'const')),
                'temperature': default(args.temperature, pp.get('temperature', 4.5)),
                'schedule_type': default(args.schedule_type, pp.get('schedule_type', 'arccos')),
                'shift': default(args.mask_ratio_shift, pp.get('mask_ratio_shift', None)),
                **logits_processor_kwargs,
            },
            guidance_scale=args.guidance_scale,
        )
        if safe_json:
            return safe_json(kwargs)
        return kwargs

    @torch.no_grad()
    def predict(self, prompt=None, label=None, **kwargs):
        """
        Predict the image from the given text.

        Args:
            prompt (str or List[str]): The input text.
            label (int): The class label of the image.
            kwargs:
                size (int): The (height, width) of the output image. Default is (256, 256).
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_sample_per_prompt (int): The number of images per prompt. Default is 1.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                prompt_embeds (torch.Tensor): Preextracted prompt embeddings.
                attention_mask (torch.Tensor): Attention mask of preextracted prompt embeddings.
                negative_prompt_embeds (torch.Tensor): Preextracted negative prompt embeddings.
                negative_attention_mask (torch.Tensor): Attention mask of preextracted negative prompt embeddings.
                infer_steps (int): The number of inference steps. Default is 100.
                verbose (int): 0 for no log, 1 for all log, 2 for fewer log. Default is 1.
                output_type (str): The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.
                    Default is 'pil'.
                pipeline_kwargs (dict): The additional arguments for the pipeline-like models.
        """
        out_dict = dict()

        # --------------------------------------------------------
        # Common arguments
        # --------------------------------------------------------
        num_sample_per_prompt = kwargs.get("num_sample_per_prompt", self.args.num_sample_per_prompt)
        negative_prompt = kwargs.pop("negative_prompt", self.args.neg_prompt)
        verbose = kwargs.get("verbose", 1)
        output_type = kwargs.get("output_type", "pil")
        infer_kwargs = self.get_infer_kwargs()
        guidance_scale = kwargs.get("guidance_scale", infer_kwargs["guidance_scale"])
        infer_steps = kwargs.get("infer_steps", infer_kwargs["infer_steps"])
        pipeline_kwargs = kwargs.get("pipeline_kwargs", infer_kwargs["pipeline_kwargs"])

        # --------------------------------------------------------
        # Prompt: label
        #         prompt, negative_prompt,
        #         prompt_embeds, attention_mask,
        #         negative_prompt_embeds, negative_attention_mask
        # --------------------------------------------------------
        if prompt is None and label is None:
            raise ValueError("Either `prompt` or `label` must be provided.")
        if prompt is not None:
            condition_dict, batch_size = self.prepare_prompts(prompt, negative_prompt, **kwargs)
            out_dict['prompt'] = condition_dict['prompt']
            out_dict['negative_prompt'] = condition_dict['negative_prompt']
        else:
            if isinstance(label, int):
                label = [label]
            batch_size = len(label)
            out_dict['label'] = label
            condition_dict = {"label": label}

        # -------------------------------------------------------
        # Inpainting: image and mask
        # -------------------------------------------------------
        if 'image' in kwargs and 'mask_image' in kwargs:
            condition_dict['image'] = kwargs['image']
            condition_dict['mask_image'] = kwargs['mask_image']

        # ---------------------------------
        # Image size
        # ---------------------------------
        size = self.parse_image_size(kwargs.get("size", (256, 256)), align=16)
        out_dict['size'] = tuple(size)

        # --------------------------------------------------------
        # Random seed: seed, num_sample_per_prompt
        # --------------------------------------------------------
        seeds = self.prepare_seed(seed=kwargs.get('seed', None),
                                  batch_size=batch_size,
                                  num_sample_per_prompt=num_sample_per_prompt,
                                  )
        out_dict['seeds'] = seeds
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        set_manual_seed(sum(seeds))

        # --------------------------------------------------------
        # Logits Processors: topK, topP
        # --------------------------------------------------------
        logits_processor_repr = ""
        if logits_processor_list := self.args.get('logits_processors_cfg', []):
            for logits_processor in logits_processor_list:
                for name, kwargs_ in logits_processor.items():
                    if name in ["TopKLogitsWarper", "TopPLogitsWarper"]:
                        for k, v in kwargs_.items():
                            logits_processor_repr += f"{k}={v}, "

        # =========================================================================
        # Inference
        # =========================================================================
        if verbose == 1:
            if prompt is None:
                info_str = f"""
                     label: {condition_dict['label']}"""
            else:
                info_str = f"""
                prompt: {condition_dict['prompt']}
            neg_prompt: {condition_dict['negative_prompt']}"""
            info_str += f"""
                  size: {size}
                  seed: {seeds}
           infer_steps: {infer_steps}
 num_sample_per_prompt: {num_sample_per_prompt}
        guidance_scale: {guidance_scale}
       pipeline_kwargs: {pipeline_kwargs}
      logits_processor: {logits_processor_repr}"""
            self.logger.info(info_str)

        start_time = time.time()
        samples = self.pipeline(**condition_dict,
                                height=size[0],
                                width=size[1],
                                num_inference_steps=infer_steps,
                                guidance_scale=guidance_scale,
                                num_sample_per_prompt=num_sample_per_prompt,
                                generator=generator,
                                output_type=output_type,
                                **pipeline_kwargs,
                                )[0]
        out_dict['samples'] = samples
        gen_time = time.time() - start_time
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")

        return out_dict


class InstructionTuningMLMSampler(Text2ImageMLMSampler):
    @classmethod
    def build_extra_model(cls, args, model_dict, factor_kwargs, logger=None):
        # The phrase `vae` is used to denote the VQ-VAE/VQ-GAN series models. It is not a VAE in the traditional sense.
        vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=factor_kwargs["device"],
            logger=logger,
        )
        model_dict["vae"] = vae

        tokenizer = TokenizerWrapper(args.tokenizer_name, logger)
        model_dict['tokenizer'] = tokenizer

        # ====================== Token Specification ======================
        text_vocab_size = model_dict["model_settings"].padded_vocab_size
        image_id_range = (-1, -1) if args.pipeline == "mar" else (text_vocab_size, text_vocab_size + vae.codebook_size)
        arrange = Arrangement(
            text_id_range=(0, text_vocab_size),
            image_id_range=image_id_range,
            pad_id=tokenizer.special_token_map["<pad>"],
            img_id=tokenizer.special_token_map["<img>"],
            mask_id=tokenizer.special_token_map["<mask>"],
            uncond_id=tokenizer.special_token_map["<cfg>"],
            boi_id=tokenizer.special_token_map["<boi>"],
            eoi_id=tokenizer.special_token_map["<eoi>"],
            s_text_range=(None, args.text_token_length - 5),        # modified
            s_text_maxlen=args.text_token_length,
            s_image_range=(args.text_token_length - 4, None),       # modified
            s_image_maxlen=-1,
            sequence=["<pad>", "<bos>", "<text>", "<boi>", "<image>", "<eoi>", "<boi>", "<image>", "<eoi>", "<eos>"],   # modified
            ignore_id=args.ignore_index,
        )
        model_dict['arrange'] = arrange

        # =========================== Build logits_processor ======================
        if args.pipeline == 'mlm':
            logits_processors = get_logits_processors(args.get('logits_processors_cfg', []))
            model_dict["logits_processor"] = logits_processors

        return model_dict

    def get_target_size(self, size_hw, anchor_size, im_hw, align=16):
        if size_hw is not None:
            th, tw = self.parse_image_size(size_hw, align=align)
        elif anchor_size is not None:
            if not hasattr(self, 'resolutions'):
                self.resolutions = {}
            if anchor_size not in self.resolutions:
                step = anchor_size // 16
                self.resolutions[anchor_size] = ResolutionGroup(anchor_size, step, align=align)
            tw, th = self.resolutions[anchor_size].get_target_size(im_hw[1], im_hw[0])
        else:
            th = im_hw[0] // 16 * 16
            tw = im_hw[1] // 16 * 16
        return th, tw

    @torch.no_grad()
    def predict(self, prompt, **kwargs):
        """
        Predict the image from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                instruction (str): The instruction for the painting task.
                size (int): The output image will have the save aspect ratio as the input image. If is int, the short
                    side will be resized to this size. If two values, the height and width of the output image.
                    Default is 256.
                anchor_size (int): The anchor size of `image`. If `size` not provided, the `image` will be resized
                    to the `anchor_size` with the same aspect ratio.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_sample_per_prompt (int): The number of images per prompt. Default is 1.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                prompt_embeds (torch.Tensor): Preextracted prompt embeddings.
                attention_mask (torch.Tensor): Attention mask of preextracted prompt embeddings.
                negative_prompt_embeds (torch.Tensor): Preextracted negative prompt embeddings.
                negative_attention_mask (torch.Tensor): Attention mask of preextracted negative prompt embeddings.
                infer_steps (int): The number of inference steps. Default is 100.
                verbose (int): 0 for no log, 1 for all log, 2 for fewer log. Default is 1.
                output_type (str): The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.
                    Default is 'pil'.
                pipeline_kwargs (dict): The additional arguments for the pipeline-like models.
        """
        out_dict = dict()

        # --------------------------------------------------------
        # Common arguments
        # --------------------------------------------------------
        guidance_scale = kwargs.get("guidance_scale", self.args.guidance_scale)
        num_sample_per_prompt = kwargs.get("num_sample_per_prompt", self.args.num_sample_per_prompt)
        negative_prompt = kwargs.pop("negative_prompt", self.args.neg_prompt)
        infer_steps = kwargs.get("infer_steps", self.args.infer_steps)
        verbose = kwargs.get("verbose", 1)
        output_type = kwargs.get("output_type", "pil")
        pipeline_kwargs = kwargs.get("pipeline_kwargs", default(self.args.pipeline_kwargs, {}))

        # --------------------------------------------------------
        # Prompt: prompt, negative_prompt,
        #         prompt_embeds, attention_mask,
        #         negative_prompt_embeds, negative_attention_mask
        #         instruction
        # --------------------------------------------------------
        condition_dict, batch_size = self.prepare_prompts(prompt, negative_prompt, **kwargs)
        out_dict['prompt'] = condition_dict['prompt']
        out_dict['negative_prompt'] = condition_dict['negative_prompt']
        if 'instruction' in kwargs:
            condition_dict['instruction'] = kwargs['instruction']

        # -------------------------------------------------------
        # Editing: image, size, mask
        # -------------------------------------------------------
        assert 'image' in kwargs or 'image_tokens', (
            "Please provide the image to be edited by `image` or `image_tokens`."
        )
        if 'image' in kwargs:
            condition_dict['image'] = kwargs['image']
            height, width = self.get_target_size(
                size_hw=kwargs.get("size"),
                anchor_size=kwargs.get("anchor_size"),
                im_hw=kwargs["image"].size[::-1],
            )   # [h, w]
        else:
            condition_dict['image_tokens'] = kwargs['image_tokens']
            tk_height, tk_width = condition_dict['image_tokens'].shape[-2:]
            height = tk_height * self.model_dict['vae'].downsample_factor
            width = tk_width * self.model_dict['vae'].downsample_factor
        out_dict['size'] = (height, width)

        if 'mask' in kwargs:
            # kwargs['mask'] is a PIL image
            condition_dict['mask'] = kwargs['mask']
        elif 'src_mask' in kwargs:
            # kwargs['src_mask'] is a torch.Tensor with down-sampled size
            condition_dict['src_mask'] = kwargs['src_mask']

        # --------------------------------------------------------
        # Random seed: seed, num_sample_per_prompt
        # --------------------------------------------------------
        seeds = self.prepare_seed(seed=kwargs.get('seed', None),
                                  batch_size=batch_size,
                                  num_sample_per_prompt=num_sample_per_prompt,
                                  )
        out_dict['seeds'] = seeds
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        set_manual_seed(sum(seeds))

        # --------------------------------------------------------
        # Logits Processors: topK, topP
        # --------------------------------------------------------
        logits_processor_repr = ""
        if logits_processor_list := self.args.get('logits_processors_cfg', []):
            for logits_processor in logits_processor_list:
                for name, kwargs_ in logits_processor.items():
                    if name in ["TopKLogitsWarper", "TopPLogitsWarper"]:
                        for k, v in kwargs_.items():
                            logits_processor_repr += f"{k}={v}, "

        # =========================================================================
        # Inference
        # =========================================================================
        if verbose == 1:
            info_str = ""
            if 'instruction' in condition_dict:
                info_str += f"""
           instruction: {condition_dict['instruction']}"""
            info_str += f"""
                prompt: {condition_dict['prompt']}
            neg_prompt: {condition_dict['negative_prompt']}
                  size: {(height, width)}
                  seed: {seeds}
           infer_steps: {infer_steps}
 num_sample_per_prompt: {num_sample_per_prompt}
        guidance_scale: {guidance_scale}
       pipeline_kwargs: {pipeline_kwargs}
      logits_processor: {logits_processor_repr}"""
            self.logger.info(info_str)

        start_time = time.time()
        samples = self.pipeline(**condition_dict,
                                height=height,
                                width=width,
                                num_inference_steps=infer_steps,
                                guidance_scale=guidance_scale,
                                num_sample_per_prompt=num_sample_per_prompt,
                                generator=generator,
                                output_type=output_type,
                                **pipeline_kwargs,
                                )[0]
        out_dict['samples'] = samples
        gen_time = time.time() - start_time
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")

        return out_dict


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)
    task = initial_args.task

    if task == "label" or task == "sample":
        auto_sampler = Text2ImageMLMSampler
    elif task == "editing" or task == "inpainting":
        auto_sampler = InstructionTuningMLMSampler
    else:
        raise ValueError(f"Invalid task: {initial_args.task}")

    sampler = auto_sampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args

    # Start evaluation
    # -----------------------------------------------------------------------------
    # Task: Label to Image (For ImageNet-1k only)
    if task == "label":
        if args.interactive and args.label is not None:
            label2image_interactive(args, sampler, logger)
        else:   # `args.n_class` required
            label2image_batch(args, "imagenet_1k", sampler, logger)

    # -----------------------------------------------------------------------------
    # Task: Text to Image
    elif task == "sample":
        if args.interactive:
            text2image_interactive(args, sampler, logger)
        else:
            text2image_batch(args=args, datasource=args.csv, sampler=sampler, logger=logger)

    # -----------------------------------------------------------------------------
    # Task: Text Guided Editing
    elif task == "editing":
        if args.interactive:
            kwargs = dict(image=args.image)
            text2image_interactive(args, sampler, logger, **kwargs)
        else:
            source = default(args.get('editing_test_arrow'), args.csv)
            if Path(source).suffix == '.arrow':
                data_source = lambda save_template: ArrowDataset(
                    arrow_file=source,
                    length=1024,
                    save_template=save_template,
                    column_dict={"seed": "seed", "prompt": "instruction",
                                 "image_tokens": "src_img_token_256x256_88-vqgan-hy_241024@ast"},
                    seed_type=args.seed_type,
                    seed=args.seed,
                    skip_exist=args.skip_exist,
                    logger=logger,
                )
                # Map dataset item keys to predict() required keys.
                input_batch_dict = dict(prompt="prompt", image_tokens="image_tokens")
            else:
                data_source = args.csv
                input_batch_dict = None
            text2image_batch(args=args, datasource=data_source, sampler=sampler, logger=logger, task=task,
                             input_batch_dict=input_batch_dict)

    # -----------------------------------------------------------------------------
    # Task: Text&Mask Guided Inpainting
    elif task == "inpainting":
        if args.interactive:
            kwargs = dict(image=args.image, mask=args.mask)
            text2image_interactive(args, sampler, logger, **kwargs)
        else:
            source = default(args.get('inpainting_test_arrow'), args.csv)
            if Path(source).suffix == '.arrow':
                data_source = lambda save_template: ArrowDataset(
                    arrow_file=source,
                    length=1024,
                    save_template=save_template,
                    column_dict={"seed": "seed", "instruction": "instruction", "prompt": "caption@json@long caption",
                                 "image_tokens": "img_token_88-vqgan-hy_241024@ast", "src_mask": "mask_token@ast"},
                    seed_type=args.seed_type,
                    seed=args.seed,
                    skip_exist=args.skip_exist,
                    logger=logger,
                )
                # Map dataset item keys to predict() required keys.
                input_batch_dict = dict(
                    instruction="instruction", prompt="prompt", image_tokens="image_tokens", src_mask="src_mask"
                )
            else:
                data_source = args.csv
                input_batch_dict = None
            text2image_batch(args=args, datasource=data_source, sampler=sampler, logger=logger, task=task,
                             input_batch_dict=input_batch_dict)

    else:
        raise ValueError(f"Invalid task: {task}")


if __name__ == "__main__":
    import os
    import ipdb
    print(f"DEBUG: {os.environ.get('DEBUG', 'false')}")
    print(f"To enable DEBUG, Set DEBUG to true in cmd  using export DEBUG=true or edit .deepspeed_env file as DEBUG=true")
    if os.environ.get("DEBUG", "false") == "true":
        print("Entering DEBUG mode and traceback will be printed")
        try:
            main()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred: {e}")
            ipdb.post_mortem()  # Enter the debugger
    else:
        main()
