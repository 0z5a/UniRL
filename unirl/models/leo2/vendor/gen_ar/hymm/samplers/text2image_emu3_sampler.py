import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
import transformers
ver = list(map(int, transformers.__version__.split('.')))
if ver[0] < 4 or (ver[0] == 4 and ver[1] < 44):
    raise ValueError("Please upgrade transformers to 4.44.0 or later.")

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM, PreTrainedModel
PreTrainedModel.generate
from transformers.generation import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.configuration_utils import GenerationConfig

from hymm.config import parse_args, parse_eval_initial_args
from hymm.constants import PRETRAINED_LLM_BASE, VAE_BASE
from hymm.data_kits.csv_dataset import CSVDataset
from hymm.data_kits.datasampler import DistributedSamplerFix
from hymm.samplers.base_sampler import setup_distributed_initialize, BaseSampler
from hymm.utils.helpers import default
from hymm.utils.eval_utils import batch_data_repr

PATH_TO_BAAI_Emu3_Stage1_MODEL = PRETRAINED_LLM_BASE + "/Emu3-Stage1"

import sys
sys.path.append(PATH_TO_BAAI_Emu3_Stage1_MODEL)
from processing_emu3 import Emu3Processor

# model path
EMU_HUB = PATH_TO_BAAI_Emu3_Stage1_MODEL
VQ_HUB = VAE_BASE + "/Emu3_VisionTokenizer"


class Emu3Stage1SamplePipeline(object):
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device

        # prepare model and processor
        logger.info(f"Loading model from {EMU_HUB}")
        self.model = AutoModelForCausalLM.from_pretrained(
            EMU_HUB,
            device_map=f"cuda:{device}",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        # prepare hyper parameters
        self.generation_config = GenerationConfig(
            use_cache=True,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=40960,
            do_sample=True,
            top_k=2048,
        )

        logger.info(f"Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
        logger.info(f"Loading image processor and tokenizer from {VQ_HUB}")
        self.image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map=f"cuda:{device}", trust_remote_code=True).eval()
        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer, chat_template="{image_prompt}{text_prompt}"
        )
        logger.info("Loaded.")

        # self.positive_prompt = " masterpiece, film grained, best quality."
        # self.negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        self.positive_prompt = ""
        self.negative_prompt = ""

        self.anchor_ratio = np.array([1, 4/3, 3/4, 9/16, 16/9])
        self.anchor_ratio_str = ["1:1", "4:3", "3:4", "16:9", "9:16"]

    def get_logits_processor(self, cfg_scale, neg_inputs, constrained_fn):
        return LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                cfg_scale,
                self.model,
                unconditional_ids=neg_inputs.input_ids.to(f"cuda:{self.device}"),
            ),
            PrefixConstrainedLogitsProcessor(
                constrained_fn,
                num_beams=1,
            ),
        ])

    def __call__(self, prompt, **kwargs):
        classifier_free_guidance = default(self.args.guidance_scale, 3.0)
        prompt += self.positive_prompt

        image_area = self.args.sample_image_size[0] * self.args.sample_image_size[1]
        ratio = self.args.sample_image_size[1] / self.args.sample_image_size[0]
        ratio_str = self.anchor_ratio_str[np.argmin(np.abs(self.anchor_ratio - ratio))]
        kwargs = dict(
            mode='G',
            ratio=ratio_str,
            image_area=image_area,
            return_tensors="pt",
            padding="longest",
        )
        pos_inputs = self.processor(text=prompt, **kwargs)
        neg_inputs = self.processor(text=self.negative_prompt, **kwargs)

        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        logits_processor = self.get_logits_processor(classifier_free_guidance, neg_inputs, constrained_fn)

        # generate
        outputs = self.model.generate(
            pos_inputs.input_ids.to(f"cuda:{self.device}"),
            self.generation_config,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(f"cuda:{self.device}"),
        )

        mm_list = self.processor.decode(outputs[0])
        images = {"samples": []}
        for idx, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            images["samples"].append(im)
        return images


@torch.no_grad()
def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    args = parse_args()

    # Create pipeline
    logger.info(f"Building Emu3Stage1SamplePipeline")
    pipeline = Emu3Stage1SamplePipeline(args, device, logger)

    def input_fn():
        if args.prompt is None:
            # Ask for the next prompt
            inputs_ = input("Input prompt (`q` to quit): ")
            if inputs_ == "q":
                return None
            prompt_ = inputs_
        else:
            prompt_ = args.prompt
            args.prompt = None
        return {'prompt': prompt_}

    if args.interactive:
        while True:
            inputs = input_fn()
            if inputs is None:
                break

            start = time.time()
            outputs = pipeline(**inputs)
            logger.info(f"Predict Time: {time.time() - start:.2f}s")
            samples = outputs["samples"]
            save_name = inputs["prompt"]
            save_paths = BaseSampler.get_default_sample_save_paths(len(samples), save_name)
            BaseSampler.save_batch_image(samples, save_paths)
            logger.info(f"Save the generated image to: {save_paths}")

    else:
        # Build sample directory with key information
        save_base = Path(args.sample_save_path)
        save_base.mkdir(parents=True, exist_ok=True)

        # Create test datasets
        logger.info(f"Building dataset")
        save_template = str(save_base / ('{}_{{}}' + f'{args.sample_save_file_suffix}.png'))
        dataset = CSVDataset(args.csv,
                             save_template=save_template,
                             seed_type=args.seed_type,
                             seed=args.seed,
                             skip_exist=args.skip_exist,
                             logger=logger,
                             )
        sampler = DistributedSamplerFix(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=args.sample_batch_size, shuffle=False, sampler=sampler, drop_last=False)
        logger.info(f"Total samples: {len(dataset.total_prompts)}")

        # Start sampling
        total_batches = len(dataloader)
        for batch_index, batch in enumerate(dataloader, start=1):
            logger.info(f"Batch {batch_index}/{total_batches}")
            # Adjust max_width according to your terminal width
            logger.info(f"\n{batch_data_repr(batch, max_width=150)}")

            # Generate samples
            final_samples = []
            save_names = []
            for i, prompt in enumerate(batch["input"]):
                outputs = pipeline(prompt)
                final_samples.extend(outputs["samples"])
                save_names.extend([
                    batch["save_path"][i].format(j)
                    for j in range(len(outputs["samples"]))
                ])

            BaseSampler.save_batch_image(final_samples, save_names)


if __name__ == "__main__":
    main()
