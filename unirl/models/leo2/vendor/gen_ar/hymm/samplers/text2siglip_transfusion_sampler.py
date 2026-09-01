import time
import math

import torch

from hymm.ar import load_pipeline
from hymm.config import parse_eval_initial_args
from hymm.diffusion import load_scheduler
from hymm.models import TokenizerWrapper
from hymm.models.basic.rope import get_3d_rope
from hymm.samplers.base_sampler import BaseSampler, setup_distributed_initialize, text2siglip_batch
from hymm.samplers.logits_processor import get_logits_processors
from hymm.utils.file_utils import rank0_logger
from hymm.utils.torch_utils import PRECISION_TO_TYPE

# Define a dummy placeholder for loading the PTM ckpt.
build_pretraining_data_loader = None


class Text2SiglipTransfusionSampler(BaseSampler):
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

        self.vision_encoder_max_num_patches = self.args.vision_encoder_max_num_patches

        pipeline_name = "siglip_transfusion"
        self.pipeline = load_pipeline(args, name=pipeline_name, rank=rank, device=device, **model_dict)

        self.tkwrapper: TokenizerWrapper = model_dict["tokenizer"]

    @staticmethod
    def build_extra_model(args, model_dict, factor_kwargs, logger=None):
        # =========================== Build text tokenizer ========================
        tokenizer = TokenizerWrapper(args.tokenizer_name, logger)
        model_dict["tokenizer"] = tokenizer

        scheduler = load_scheduler(args)
        model_dict["scheduler"] = scheduler

        # =========================== Build logits_processor ======================
        if args.do_sample:
            logits_processors = get_logits_processors(args.get('logits_processors_cfg', []))
            model_dict["logits_processor"] = logits_processors

        return model_dict

    def get_infer_kwargs(self, safe=False):
        args = self.args
        kwargs = dict(
            diff_infer_steps=args.diff_infer_steps,
            guidance_scale=args.guidance_scale,
        )
        return kwargs

    @torch.no_grad()
    def predict(self, prompt, **kwargs):
        out_dict = {}

        guidance_scale = kwargs.get("guidance_scale", self.args.guidance_scale)
        diff_infer_steps = kwargs.get("diff_infer_steps", self.args.diff_infer_steps)
        flow_shift = kwargs.get("flow_shift", self.args.sample_flow_shift)
        output_type = kwargs.get("output_type", "pil")
        verbose = kwargs.get("verbose", 1)
        use_recaption_template = kwargs.get("use_caption_template", self.args.get('use_recaption_template', False))
        uncond_enabled = kwargs.get("uncond_enabled")

        if use_recaption_template:
            captions, _ = self.recaption(prompt, seed=kwargs.get('seed'), verbose=verbose)
            prompt, _, _ = self.prepare_template(captions, task="recap_gen_image", prompts=prompt)
            uncond_enabled = [True, False, True, False]  # [short] <recaption> [long] </recaption>

        condition_dict, batch_size = self.prepare_prompts(prompt, **kwargs)
        prompt = condition_dict['prompt']
        out_dict["prompts"] = prompt

        seeds = self.prepare_seed(
            seed=kwargs.get('seed'),
            batch_size=batch_size,
            num_sample_per_prompt=1,
        )
        out_dict["seeds"] = seeds
        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        image_size = self.parse_image_size(kwargs["size"], align=16)
        h, w = image_size[0], image_size[1]

        tk_height = h // int(math.sqrt(self.vision_encoder_max_num_patches))
        tk_width = w // int(math.sqrt(self.vision_encoder_max_num_patches))

        actual_image_token_length = self.vision_encoder_max_num_patches

        image_token_shape_wh = torch.tensor([tk_width, tk_height], dtype=torch.long).unsqueeze(0).to(self.device)
        iw_ih_scatter_src = image_token_shape_wh.repeat(batch_size, 1)

        tokenizer: TokenizerWrapper = self.model_dict["tokenizer"]
        tokens, iw_ih_scatter_index, timestep_scatter_index, _, image_mask = tokenizer.batch_gen_infer(
            infer_fn=tokenizer.encode_transfusion,
            prompt_list=prompt,
            infer_fn_kwargs_list=[dict(
                image_token_length=actual_image_token_length,
                max_text_token_length=self.args.text_token_length + 1,
                max_image_token_length=actual_image_token_length,
                uncond_enabled=uncond_enabled,
                add_iw_ih_token=self.args.add_iw_ih_token,
                add_timestep_token=self.args.add_timestep_token,
                use_front_boi_token=self.args.use_front_boi_token,
            ) for _ in range(batch_size)],
            do_classifier_free_guidance=guidance_scale > 1.0,
        )

        tokens = tokens[:, :-1].contiguous().to(self.device)
        image_mask = image_mask[:, :-1].contiguous().to(self.device)

        # build attention mask
        n_tokens = tokens.shape[1]
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=self.device).tril(diagonal=0)
        causal_mask = causal_mask.view(1, n_tokens, n_tokens).repeat(tokens.shape[0], 1, 1)
        image_mask_1 = image_mask.view(tokens.shape[0], 1, n_tokens).repeat(1, n_tokens, 1)
        image_mask_2 = image_mask_1.transpose(1, 2)
        attention_mask = causal_mask | (image_mask_1.bool() & image_mask_2.bool())
        # unsqueeze for attention head dim
        attention_mask = attention_mask.unsqueeze(1)

        freqs_cos, freqs_sin = None, None

        # repeat for cfg
        if guidance_scale > 1.0:
            iw_ih_scatter_src = iw_ih_scatter_src.repeat(2, 1)

        # with cond and uncond
        model_input_extra_kwargs = dict(
            idx=tokens,  # [b, 512]
            image_mask=image_mask,    # [b, 512]
            attention_mask=attention_mask,  # [b, 512, 512]
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )
        if iw_ih_scatter_index is not None:
            model_input_extra_kwargs.update({
                "iw_ih_scatter_index": iw_ih_scatter_index.to(self.device),  # [b, 2]
                "iw_ih_scatter_src": iw_ih_scatter_src.to(self.device),  # [b, 2]
            })
        if timestep_scatter_index is not None:
            model_input_extra_kwargs.update({
                "timestep_scatter_index": timestep_scatter_index.to(self.device),  # [b, 1]
            })

        # flow shift
        scheduler_set_timesteps_extra_kwargs = {}
        if self.args.sample_use_flux_shift:
            scheduler_set_timesteps_extra_kwargs['n_tokens'] = actual_image_token_length

        if verbose == 1:
            info_str = f"""
            prompt: {condition_dict['prompt']}
            """
            info_str += f"""
            image_size: {image_size}
                  seed: {seeds}
      diff_infer_steps: {diff_infer_steps}
        guidance_scale: {guidance_scale}
            flow_shift: {flow_shift}
          extra_kwargs: {self.get_infer_kwargs()}
            """
            self.logger.info(info_str)

        start_time = time.time()
        latents = self.pipeline(batch_size=batch_size,
                                image_size=image_size,
                                num_inference_steps=diff_infer_steps,
                                guidance_scale=guidance_scale,
                                generator=generators,
                                output_type=output_type,
                                model_input_extra_kwargs=model_input_extra_kwargs,
                                scheduler_set_timesteps_extra_kwargs=scheduler_set_timesteps_extra_kwargs,
                                )[0]
        out_dict['latents'] = latents
        gen_time = time.time() - start_time
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")

        return out_dict


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)

    sampler = Text2SiglipTransfusionSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args


    if args.csv:
        segments = [f"{args.denoise_type}{args.diff_infer_steps}", f"cfg{args.guidance_scale}", f"shift{args.sample_flow_shift}"]
        text2siglip_batch(
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
