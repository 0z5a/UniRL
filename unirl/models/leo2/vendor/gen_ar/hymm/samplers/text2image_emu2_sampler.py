import time

import torch
from PIL import Image
import numpy as np

from hymm.ar import load_pipeline
from hymm.config import parse_eval_initial_args
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

import cv2
from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Text2ImageEMU2Sampler(BaseSampler):
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
        if logger is None:
            from loguru import logger
        logger.info("Loading multimodal_encoder ...")

        # if model_dict.get('arrange', None) is None:
        #     pipeline_name = "maskgit"
        # else:
        #     pipeline_name = default(args.pipeline, "mlm")
        # self.pipeline = load_pipeline(args, name=pipeline_name, rank=rank, device=device, **model_dict)

        path = "/apdcephfs_sh7/share_301124792/chenyangqi/pretrained_ckpt/BAAI/Emu2-Gen"
        # path = "/apdcephfs_cq8/share_2938211/chenyangqi/repos/Emu/ckpt_cq8/Emu2-Gen"
        logger.info(f"Loading multimodal_encoder from {path}")
        multimodal_encoder = AutoModelForCausalLM.from_pretrained(
            f"{path}/multimodal_encoder",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="bf16"
        )
        logger.info(f"Moving multimodal_encoder to cuda {device}")
        multimodal_encoder = multimodal_encoder.to(device)
        logger.info(f"Loading tokenizer from {path}/tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")
        logger.info(f"Loading pipe from {path}")
        pipeline = DiffusionPipeline.from_pretrained(
            path,
            custom_pipeline="pipeline_emu2_gen",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="bf16",
            multimodal_encoder=multimodal_encoder,
            tokenizer=tokenizer,
        )
        logger.info(f"Moving pipe to cuda {device}")
        self.pipeline = pipeline.to(device)
        logger.info(f"Pipeline loaded and moved to cuda {device}")



    @classmethod
    def build_extra_model(cls, args, model_dict, factor_kwargs, logger=None):


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
        guidance_scale = kwargs.get("guidance_scale", self.args.guidance_scale)
        num_sample_per_prompt = kwargs.get("num_sample_per_prompt", self.args.num_sample_per_prompt)
        negative_prompt = kwargs.pop("negative_prompt", self.args.neg_prompt)
        infer_steps = kwargs.get("infer_steps", self.args.infer_steps)
        verbose = kwargs.get("verbose", 1)
        output_type = kwargs.get("output_type", "pil")
        pipeline_kwargs = kwargs.get("pipeline_kwargs", default(self.args.pipeline_kwargs, {}))

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

        samples = self.pipeline(prompt)
        
        #######################
        images_numpy = samples.images_numpy
        images_numpy: np.ndarray
        if len(images_numpy.shape) == 3:
            out_dict_samples = images_numpy[None, ...]
        else:
            out_dict_samples = images_numpy
        ##############
        images_torch = samples.images_torch
        images_torch: torch.Tensor
        if len(images_torch.shape) == 3:
            out_dict_samples = images_torch[None, ...]
        else:
            out_dict_samples = images_torch
        ##############
        
        # out_dict['samples'] = samples
        out_dict['samples'] = out_dict_samples
        gen_time = time.time() - start_time
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")

        return out_dict



def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)

    print(f"Launch sampler with mode: {mode}")
    print(f"Distributed environment: world_size: {world_size}, rank: {rank}, device: {device}")

    logger = rank0_logger(rank)

    sampler = Text2ImageEMU2Sampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args

    # Start sampling or evaluation
    if args.evaluation_metrics:
        logger.info(f"initialize_scores based on evaluation_metrics: {args.evaluation_metrics}")
        sampler.initialize_scores(args.evaluation_metrics)

        save_path = sampler.get_metric_save_path(args.metric_image_size, "score")
        logger.info(f"save_path: {save_path}")
        image_save_base = (
            sampler.get_sample_save_dir(testset=None, image_size=args.metric_image_size)
            if args.evaluation_metrics_save_image
            else None
        )
        logger.info(f"image_save_base: {image_save_base}")
        kwargs = sampler.get_infer_kwargs()

        sampler.eval(
            image_size=args.metric_image_size,
            batch_size=args.metric_batch_size,
            save_path=save_path,
            image_save_base=image_save_base,
            extra_save_info=safe_json(kwargs),
            **kwargs,
        )

    elif args.interactive:
        text2image_interactive(args, sampler, logger)

    elif args.csv is not None:
        text2image_batch(args, args.csv, sampler, logger)

    else:
        raise ValueError("Please provide `--evaluation-metrics`, `--interactive` (with `--prompt`), or `--csv`.")


if __name__ == "__main__":
    import ipdb
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
        ipdb.post_mortem()  # Enter the debugger
