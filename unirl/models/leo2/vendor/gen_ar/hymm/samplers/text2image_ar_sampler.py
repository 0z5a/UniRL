from einops import rearrange

import torch

from hymm.samplers.base_sampler import BaseSampler, setup_distributed_initialize, text2image_interactive, text2image_batch
from hymm.samplers.logits_processor import get_logits_processors
from hymm.config import parse_eval_initial_args
from hymm.models import load_vae, TokenizerWrapper
from hymm.utils.file_utils import rank0_logger
from hymm.utils.torch_utils import PRECISION_TO_TYPE


class Text2ImageARSampler(BaseSampler):
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

        return model_dict

    @torch.no_grad()
    def get_next_token(self, x, input_pos, generators, no_iw_ih_scatter, real_pos, **model_intput_kwargs):
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_enabled):
            outs = self.model_dict["model"](x, input_pos=input_pos, no_iw_ih_scatter=no_iw_ih_scatter, **model_intput_kwargs)

        if not self.enable_sep_modal_adaptors:
            logits = outs["logits"]
        else:
            logits = outs["image_logits"]

        logits_dim = logits.shape[-1]
        if logits.shape[1] > 1:
            logits = logits.gather(index=real_pos.unsqueeze(-1).expand(-1, -1, logits_dim), dim=1).squeeze(1)
        else:
            logits = logits[:, -1, :]
        logits = self.model_dict["logits_processor"](x, logits)
        if self.cfg_enabled:
            logits, _ = torch.chunk(logits, chunks=2, dim=0)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token_list = []
        for batch_idx, generator in enumerate(generators):
            next_token_list.append(torch.multinomial(probs[batch_idx], num_samples=1, generator=generator))
        next_token = torch.cat(next_token_list, dim=0).unsqueeze(-1)

        if self.cfg_enabled:
            next_token = next_token.repeat((2, 1))
        return next_token
    
    def format_prompt(self, prompt, format_type="default"):
        if format_type == "janus":
            formatted_prompt = f"User: {prompt}\n\nAssistant:"
        else:
            formatted_prompt = prompt
        return formatted_prompt
    
    @torch.no_grad()
    def predict(self, prompt, **kwargs):
        format_prompt_type = self.args.get("format_prompt_type", "default")
        prompt = self.format_prompt(prompt, format_prompt_type)
        condition_dict, batch_size = self.prepare_prompts(prompt, negative_prompt=None, **kwargs)
        prompt = condition_dict['prompt']

        seeds = self.prepare_seed(
            seed=kwargs.get('seed', None),
            batch_size=batch_size,
            num_sample_per_prompt=1,
        )

        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        image_size = self.parse_image_size(kwargs["size"])
        latent_h = int(image_size[0] / self.vae_downsample_factor[0])
        latent_w = int(image_size[1] / self.vae_downsample_factor[1])
        image_seq_len = latent_h * latent_w

        tokenizer = self.model_dict["tokenizer"]
        # 'encode_ar_text_only_infer' is used for janus-like infer
        tokenizer_infer_fn = self.args.get("tokenizer_infer_fn", "defualt")
        if tokenizer_infer_fn == "encode_ar_text_only_infer":
            infer_fn = tokenizer.encode_ar_text_only_infer
            infer_fn_kwargs_list = [{} for _ in range(batch_size)]
        else:
            infer_fn = tokenizer.encode_ar_t2i_infer
            infer_fn_kwargs_list = [dict(
                add_iw_ih_token=self.args.add_iw_ih_token,
                use_front_boi_token=self.args.use_front_boi_token,
            ) for _ in range(batch_size)]
        input_ids, iw_ih_scatter_index, batched_real_pos = tokenizer.batch_gen_infer(
            infer_fn=infer_fn,
            prompt_list=prompt,
            infer_fn_kwargs_list=infer_fn_kwargs_list,
            do_classifier_free_guidance=self.args.guidance_scale > 1.0,
        )
        input_ids = input_ids.to(self.device)
        batched_real_pos = batched_real_pos.to(self.device)

        iw_ih_scatter_src = None
        if iw_ih_scatter_index is not None:
            iw_ih_scatter_index = iw_ih_scatter_index.to(self.device)
            iw_ih_scatter_src = torch.tensor([latent_w, latent_h], dtype=torch.long, device=self.device)
            iw_ih_scatter_src = iw_ih_scatter_src.unsqueeze(0).repeat(input_ids.shape[0], 1)

        model_intput_kwargs = {
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": iw_ih_scatter_src,
        }

        image_token_ids = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=self.device)
        input_pos = torch.arange(0, input_ids.shape[1], device=self.device, dtype=torch.long)

        current_pos = batched_real_pos
        self.model_dict["model"].set_kv_cache(batch_size=input_ids.shape[0], device=self.device)
        for i in range(image_seq_len):
            img_token = self.get_next_token(
                input_ids,
                input_pos,
                generators=generators,
                no_iw_ih_scatter=(i!=0),
                real_pos=current_pos-1,
                **model_intput_kwargs
            )
            next_token = img_token + self.image_token_offset
            image_token_ids = torch.cat([image_token_ids, img_token], dim=-1)
            input_ids = next_token
            # print(next_token)
            # input_pos = torch.tensor([current_pos], device=self.device, dtype=torch.long)
            input_pos = current_pos.clone()
            current_pos += 1
        # must clear kv cache in interactive mode
        self.model_dict["model"].clear_kv_cache()
        if self.cfg_enabled:
            image_token_ids, _ = torch.chunk(image_token_ids, chunks=2, dim=0)
        image_token_ids = rearrange(image_token_ids, "b (h w) -> b h w", h=latent_h, w=latent_w)
        with torch.autocast(device_type="cuda", dtype=self.vae_dtype, enabled=self.vae_dtype != torch.float32):
            results_img = self.model_dict["vae"].vq_decode(image_token_ids)
        if self.vae_trans_type == "-11":
            results_img = results_img.clamp(-1, 1) * 0.5 + 0.5
        elif self.vae_trans_type == "01":
            results_img = results_img.clamp(0, 1)
        return {"samples": results_img}


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)

    sampler = Text2ImageARSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args

    # Start evaluation
    if args.interactive:
        text2image_interactive(args, sampler, logger)

    elif args.csv:
        # add logits processor config into segments automatically, cfg is included
        segments = []
        logits_processor_list = args.get("logits_processors_cfg", [])
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
