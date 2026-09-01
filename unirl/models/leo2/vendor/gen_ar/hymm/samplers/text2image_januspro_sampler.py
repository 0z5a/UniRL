import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from hymm.config import parse_args, parse_eval_initial_args
from hymm.data_kits.csv_dataset import CSVDataset
from hymm.data_kits.datasampler import DistributedSamplerFix
from hymm.samplers.base_sampler import setup_distributed_initialize, BaseSampler
from hymm.utils.eval_utils import batch_data_repr
from hymm.utils.file_utils import rank0_logger
from janus.models import MultiModalityCausalLM, VLChatProcessor


class JanusProSamplePipeline(BaseSampler):
    def __init__(self, args, model_path, rank, world_size, device, logger):
        super().__init__(args, dict(), rank=rank, world_size=world_size, device=device, logger=logger)
        self.args = args
        self.rank = rank
        self.device = device

        # prepare model and processor
        logger.info(f"Loading VLChatProcessor from {model_path}")
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        logger.info(f"Loading MultiModalityCausalLM from {model_path}")
        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).cuda().eval()
        logger.info("Loaded.")

    @torch.inference_mode()
    def predict(self, prompt, **kwargs):
        if isinstance(prompt, list):
            assert len(prompt) == 1, "Only support one prompt for now."
            prompt = prompt[0]

        # Extra arguments
        temperature = kwargs.get("temperature", 1)
        parallel_size = kwargs.get("parallel_size", 1)
        cfg_weight = kwargs.get("guidance_scale", 5)
        image_token_num_per_image = kwargs.get("image_token_num_per_image", 576)
        img_size = kwargs.get("size", 384)
        patch_size = kwargs.get("patch_size", 16)
        output_type = kwargs.get("output_type", "pil")

        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag

        # Prompt to input_ids
        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        # ids to embeds
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        # generate image tokens
        pbar = range(image_token_num_per_image)
        if self.rank == 0:
            pbar = tqdm(pbar, desc="Generating image tokens")
        for i in pbar:
            outputs = self.model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True,
                                                      past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                      shape=[parallel_size, 8, img_size // patch_size,
                                                             img_size // patch_size])
        dec = dec.to(torch.float32)
        if output_type == "pil":
            dec = dec.cpu().numpy().transpose(0, 2, 3, 1)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)

            visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec

            images = {"samples": []}
            for idx, im in enumerate(visual_img):
                im = Image.fromarray(im)
                images["samples"].append(im)

        elif output_type == "pt":
            # Denormalize
            dec = torch.clamp((dec + 1) / 2, 0, 1)
            images = {"samples": dec}

        else:
            raise ValueError(f"Unknown output type: {output_type}")

        return images


@torch.no_grad()
def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    args = parse_args()
    logger = rank0_logger(rank)

    # Create pipeline
    logger.info(f"Building JanusProSamplePipeline")
    model_path = "/apdcephfs_gy2/share_302507476/1_public_models/hymm_ar_assets/pretrained_llm/Janus-Pro-7B"
    pipeline = JanusProSamplePipeline(args, model_path, rank, world_size, device, logger)

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
            outputs = pipeline.predict(**inputs)
            logger.info(f"Predict Time: {time.time() - start:.2f}s")
            samples = outputs["samples"]
            save_name = inputs["prompt"]
            save_paths = BaseSampler.get_default_sample_save_paths(len(samples), save_name)
            BaseSampler.save_batch_image(samples, save_paths)
            logger.info(f"Save the generated image to: {save_paths}")

    elif args.csv:
        # Build sample directory with key information
        save_base = Path(args.sample_save_path)
        save_base = save_base.parent / f"{save_base.name}{args.sample_save_path_suffix}"
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
                outputs = pipeline.predict(prompt)
                final_samples.extend(outputs["samples"])
                save_names.extend([
                    batch["save_path"][i].format(j)
                    for j in range(len(outputs["samples"]))
                ])

            BaseSampler.save_batch_image(final_samples, save_names)

    elif args.evaluation_metrics:
        # Build sample directory with key information
        save_file = Path(args.metric_save_path)
        save_file = save_file.parent / f"{save_file.name}{args.metric_save_file_suffix}"
        save_file.parent.mkdir(parents=True, exist_ok=True)

        pipeline.initialize_scores(args.evaluation_metrics)
        pipeline.load_score_models(size=384)
        pipeline.eval(
            image_size=384,
            batch_size=1,
            save_path=save_file,
            image_save_base="januspro_t2i_compbench_images"
        )


if __name__ == "__main__":
    main()
