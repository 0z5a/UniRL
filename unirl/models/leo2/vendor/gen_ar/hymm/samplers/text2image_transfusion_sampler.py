import time

import torch

from hymm.ar import load_pipeline
from hymm.config import parse_eval_initial_args
from hymm.diffusion import load_scheduler
from hymm.models import load_vae, TokenizerWrapper
from hymm.models.basic.rope import get_3d_rope
from hymm.parallelism.parallel_states import get_parallel_state
from hymm.samplers.base_sampler import BaseSampler, setup_distributed_initialize, text2image_interactive, \
    text2image_batch, setup_ptm_initialize
from hymm.samplers.logits_processor import get_logits_processors
from hymm.utils.file_utils import rank0_logger
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.utils.resolution import ResolutionGroup

# Define a dummy placeholder for loading the PTM ckpt.
build_pretraining_data_loader = None


class Text2ImageTransfusionSampler(BaseSampler):
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
        
        self.vae_downsample_factor = model_dict["vae"]._downsample_factor
        self.vae_trans_type = model_dict["vae"]._trans_type

        pipeline_name = "transfusion"
        self.pipeline = load_pipeline(args, name=pipeline_name, rank=rank, device=device, **model_dict)

        self.use_3d_rope = args.get('rope_type', 'default') in ['3d', '3d-interleave']
        self.tkwrapper: TokenizerWrapper = model_dict["tokenizer"]

        if args.add_image_shape_token:
            self.reso_group = ResolutionGroup(base_size=args.training_image_size)

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

        patch_size = self.args.patch_size
        image_size = self.parse_image_size(kwargs["size"], align=[self.vae_downsample_factor[0] * patch_size, self.vae_downsample_factor[1] * patch_size])
        h, w = image_size[0], image_size[1]
        tk_height = h // (self.vae_downsample_factor[0] * patch_size)
        tk_width = w // (self.vae_downsample_factor[1] * patch_size)
        actual_image_token_length = tk_height * tk_width

        if self.args.add_image_shape_token:
            base_size, ratio_idx = self.reso_group.get_base_size_and_ratio_index(w, h)
            image_sections = [dict(length=actual_image_token_length,
                                   base_size=base_size, ratio_idx=ratio_idx)]
        elif self.args.add_timestep_token:
            image_token_shape_wh = torch.tensor([w, h], dtype=torch.long).unsqueeze(0).to(self.device)
            iw_ih_scatter_src = image_token_shape_wh.repeat(batch_size, 1)
            image_sections = None
        else:
            raise NotImplementedError()

        tokenizer: TokenizerWrapper = self.model_dict["tokenizer"]
        tokens, iw_ih_scatter_index, timestep_scatter_index, _, image_mask = tokenizer.batch_gen_infer(
            infer_fn=tokenizer.encode_transfusion,
            prompt_list=prompt,
            infer_fn_kwargs_list=[dict(
                image_token_length=actual_image_token_length,
                max_text_token_length=self.args.text_token_length + 1,
                max_image_token_length=actual_image_token_length,
                uncond_enabled=uncond_enabled,
                add_iw_ih_token=self.args.add_iw_ih_token if not self.use_3d_rope else False,
                add_timestep_token=self.args.add_timestep_token,
                use_front_boi_token=self.args.use_front_boi_token,
                add_image_shape_token=self.args.add_image_shape_token,
                image_sections=image_sections, # if image_sections is not None, image_token_length are ignored
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

        # Prepare RoPE freqs
        if self.use_3d_rope:
            freqs_cos_list, freqs_sin_list = [], []
            for i in range(batch_size):
                img_pos = torch.where(tokens[i] == tokenizer.special_token_map["<img>"])[0][0].item()
                freqs_cos, freqs_sin = get_3d_rope(
                    self.args.rope_dim_list,
                    tk_height, tk_width,
                    max_len=self.args.text_token_length + 1,
                    img_pos=img_pos,
                    device=self.device,
                    theta=self.args.get('rope_theta', 10000),
                    use_real=True,
                    theta_rescale_factor=self.args.rope_base_rescale_factor,
                    interleave=self.args.rope_type == "3d-interleave",
                )
                freqs_cos_list.append(freqs_cos)
                freqs_sin_list.append(freqs_sin)
            freqs_cos = torch.stack(freqs_cos_list, dim=0)
            freqs_sin = torch.stack(freqs_sin_list, dim=0)
            if guidance_scale > 1.0:
                freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=0)
                freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=0)
        else:
            freqs_cos, freqs_sin = None, None

        # repeat for cfg
        if iw_ih_scatter_index is not None and guidance_scale > 1.0:
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
        samples = self.pipeline(batch_size=batch_size,
                                image_size=image_size,
                                num_inference_steps=diff_infer_steps,
                                guidance_scale=guidance_scale,
                                generator=generators,
                                output_type=output_type,
                                model_input_extra_kwargs=model_input_extra_kwargs,
                                scheduler_set_timesteps_extra_kwargs=scheduler_set_timesteps_extra_kwargs,
                                )[0]
        out_dict['samples'] = samples
        gen_time = time.time() - start_time
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")

        return out_dict


def main():
    initial_args, mode = parse_eval_initial_args()

    # ddp or deepspeed inference
    if mode != 'ptm':
        world_size, rank, device = setup_distributed_initialize(initial_args, mode)
        logger = rank0_logger(rank)


        # Initialize TP distributed environment outside the Sampler
        if initial_args.tp_size > 1:
            from torch import distributed as dist
            from torch.distributed.device_mesh import init_device_mesh
            tp_size = initial_args.tp_size
            device_mesh = init_device_mesh('cuda', (dist.get_world_size() // tp_size, tp_size), mesh_dim_names=('dp', 'tp'))
            rank = device_mesh['dp'].get_local_rank()
            world_size = device_mesh['dp'].size()

        assert rank == get_parallel_state().dp_mesh.get_local_rank()
        assert world_size == get_parallel_state().dp_mesh.size()
        sampler = Text2ImageTransfusionSampler.from_pretrained(
            ckpt_path=initial_args.ckpt,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )
        # Get updated args (include the yaml configs saved along with model checkpoint)
        args = sampler.args

        # if initial_args.tp_size > 1:
        #     from hymm.trainers import transfusion_parallel
        #     from hymm.models.autoregressive.transfusion import Transfusion
        #     from hymm.models.autoregressive.multimodal_transfusion import MultiModalTransfusion
        #
        #     model = sampler.pipeline.model
        #
        #     assert isinstance(model, (Transfusion, MultiModalTransfusion, ))
        #     if isinstance(model, MultiModalTransfusion):
        #         transfusion_parallel.apply_tp(model.language_model, device_mesh["tp"])
        #     else:
        #         transfusion_parallel.apply_tp(model, device_mesh["tp"])
        if initial_args.ep_size > 1 or initial_args.pp_size > 1:
            from hymm.parallelism.engines.gemini_parallel import GeminiParallelEngine
            model = sampler.pipeline.model
            model = GeminiParallelEngine(
                model=model,
                load_ckpt_path=initial_args.puretorch_ckpt if initial_args.puretorch_ckpt else None,
                micro_batch_size=2,
                pp_enable_autocast=True,
                # pp_enable_autocast=False,
                # weight_prec=self.args.precision,
                weight_prec='bf16',
                # cpu_offload=True,
            )
            model.eval()
            sampler.pipeline.model = model
            sampler.model_dict['model'] = model
            sampler.model = model
            model.reshard()

            from torch import distributed as dist
            import loguru
            local_rank = dist.get_node_local_rank()
            mem_t = torch.cuda.get_device_properties(local_rank).total_memory
            mem_r = torch.cuda.memory_reserved(local_rank)
            mem_a = torch.cuda.memory_allocated(local_rank)
            mem_fir = mem_r - mem_a  # free inside reserved
            mem_f = mem_t - mem_a
            dist.get_node_local_rank()
            loguru.logger.info(f'Model initialized reserved_memory={mem_r/2**30:.2f}G allocated_memory={mem_a/1e9:.2f}G res_lef={mem_fir/2**30:.2f}G free={mem_f/2**30:.2f}G')
            from accelerate.utils import set_seed
            set_seed(get_parallel_state().dp_mesh.get_local_rank())

    # ptm inference
    else:
        # use extra_args to init megatron
        args = setup_ptm_initialize()
        logger = rank0_logger(torch.distributed.get_rank())
        sampler = Text2ImageTransfusionSampler.ptm_from_pretrained(args, logger)

    # Start evaluation
    if args.interactive:
        text2image_interactive(args, sampler, logger)

    elif args.csv:
        segments = [f"{args.denoise_type}{args.diff_infer_steps}", f"cfg{args.guidance_scale}"]
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
