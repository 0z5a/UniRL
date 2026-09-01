import time
from typing import Optional, List, Union, Dict, Any
from pathlib import Path
from functools import partial

import math
import easydict
import torch
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

from hymm.ar import load_pipeline
from hymm.ar.image_processors import DefaultImageProcessor, VAEImageProcessor, FaceImageProcessor
from hymm.config import parse_eval_initial_args
from hymm.constants import VISION_ENCODER_META_INFO
from hymm.data_kits.arrow_dataset import ArrowDataset
from hymm.diffusion import load_scheduler, load_denoiser
from hymm.models import load_vae, TokenizerWrapper
from hymm.models.visual_encoders import load_insight_face, load_vision_model_processor
from hymm.samplers.base_sampler import BaseSampler, setup_distributed_initialize, text2image_interactive, \
    text2image_batch, image2x_interactive, lm_interactive
from hymm.samplers.logits_processor import get_logits_processors
from hymm.utils.eval_utils import batch_data_repr
from hymm.utils.file_utils import rank0_logger
from hymm.utils.helpers import default, to_2tuple
from hymm.utils.torch_utils import PRECISION_TO_TYPE


# Define a dummy placeholder for loading the PTM ckpt.
def build_pretraining_data_loader():
    pass


# When enabling --weights-only in pytorch>=2.4, we must manually allow the deserialization of EasyDict and set.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([set, easydict.EasyDict])


class MultimodalTransfusionSampler(BaseSampler):
    def __init__(self, args, model_dict, ckpt_path=None, rank=0, world_size=1, device=0, pipeline_name="transfusion", logger=None):
        super().__init__(
            args=args,
            model_dict=model_dict,
            ckpt_path=ckpt_path,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )
        # VAE and src/gen image processor
        self.vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
        self.vae_downsample_factor = model_dict["vae"]._downsample_factor
        self.vae_trans_type = model_dict["vae"]._trans_type
        self.vae_processor: VAEImageProcessor = model_dict["vae_processor"]

        # Vision encoder and und image processor
        self.has_vision_encoder = hasattr(args, "vision_model_type") and args.vision_model_type is not None
        if self.has_vision_encoder:
            self.vision_encoder_meta_info = VISION_ENCODER_META_INFO[args.vision_model_type]
            if args.vision_model_type == "siglip2-so400m-patch16-naflex":
                self.vision_encoder_processor = load_vision_model_processor(args.vision_model_type)
                self.vision_encoder_h_factor, self.vision_encoder_w_factor = to_2tuple(self.vision_encoder_meta_info["downsample_factor"])
                self.vision_encoder_image_token_length = args.vision_encoder_max_num_patches
                self.vision_encoder_processor = partial(self.vision_encoder_processor, max_num_patches=self.vision_encoder_image_token_length)
            else:
                self.vision_downsample_factor = VISION_ENCODER_META_INFO[args.vision_model_type]["downsample_factor"]
                self.vision_processor = model_dict["image_processor"]
                self.vision_encoder_image_size = to_2tuple(VISION_ENCODER_META_INFO[args.vision_model_type]["image_size"])
                self.vision_encoder_image_token_length = math.prod(self.vision_encoder_image_size) // math.prod(self.vision_downsample_factor)
                self.pad_color = (127, 127, 127)
                self.vision_encoder_transform = transforms.Compose([
                    transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                    transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
                ])

        self.pipeline = load_pipeline(args, name=pipeline_name, rank=rank, device=device,
                                      progress_bar_config=dict(leave=False, disable=self.rank != 0), **model_dict)

        self.denoiser = load_denoiser(args)

        self.tkwrapper: TokenizerWrapper = model_dict["tokenizer"]
        self.conv_format = "hunyuan-gemini-alpha"

        self.__post_init__()

    def __post_init__(self):
        pass

    @staticmethod
    def build_extra_model(args, model_dict, factor_kwargs, logger=None):
        if "vae" not in model_dict or model_dict["vae"] is None:
            vae = load_vae(
                args.vae_type,
                args.vae_precision,
                device=factor_kwargs["device"],
                logger=logger,
                weights_only=args.weights_only,
            )
            model_dict["vae"] = vae
        model_dict["vae_processor"] = VAEImageProcessor(
            base_size=args.training_image_size,
            vae_trans_type=model_dict["vae"]._trans_type
        )

        # =========================== Build text tokenizer ========================
        tokenizer = TokenizerWrapper(args.tokenizer_name, logger)
        model_dict["tokenizer"] = tokenizer

        scheduler = load_scheduler(args)
        model_dict["scheduler"] = scheduler

        # =========================== Build logits_processor ======================
        if args.do_sample:
            logits_processors = get_logits_processors(args.get('logits_processors_cfg', []))
            model_dict["logits_processor"] = logits_processors

        # =========================== Build image processor ======================
        if hasattr(args, "vision_model_type") and args.vision_model_type is not None:
            if args.vision_model_type == "siglip2-so400m-patch16-naflex":
                model_dict["image_processor"] = load_vision_model_processor(args.vision_model_type)
            else:
                model_dict["image_processor"] = DefaultImageProcessor(
                    image_size=VISION_ENCODER_META_INFO[args.vision_model_type]["image_size"])
        
        # =========================== Build face analysis ======================
        if hasattr(args, "load_insightface") and args.load_insightface:
            model_dict["face_analysis"] = load_insight_face()
            model_dict["face_image_processor"] = FaceImageProcessor()
        return model_dict

    def vae_encode(self, image, sample_type=None, n_tokens=None, cfg_factor=1):
        vae = self.model_dict['vae']
        # ===================================== prepare diffusion =====================================
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            vae_encode_result = vae.encode(image)
            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            else:
                latents = vae_encode_result.latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor:
                latents.sub_(vae.config.shift_factor)
            if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor:
                latents.mul_(vae.config.scaling_factor)

        # b c t h w
        if hasattr(vae, "ffactor_temporal"):
            assert latents.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
            latents = latents.squeeze(2)

        if sample_type is not None:
            x_t = latents
            model_t = torch.ones((latents.shape[0],)) * (0 if self.args.flow_reverse else 1) * 1000

            # Only for inference
            if cfg_factor > 1:
                model_t = model_t.repeat(cfg_factor)
                x_t = x_t.repeat(cfg_factor, 1, 1, 1)

            return model_t, x_t

        return latents

    def get_infer_kwargs(self, safe=False):
        args = self.args
        kwargs = dict(
            diff_infer_steps=args.diff_infer_steps,
            guidance_scale=args.guidance_scale,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        return kwargs

    @torch.no_grad()
    def get_next_token(self, x, input_pos, generators, real_pos, logits_processor=None, return_logits=False,
                       add_dummy_tokens=0, penalty=1.0, next_token_ids=None, do_sample=True, **model_intput_kwargs):
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_enabled):
            if add_dummy_tokens == 0:
                outs = self.model_dict["model"](x, input_pos=input_pos, **model_intput_kwargs)
            else:
                x_1 = torch.cat([x, torch.full((x.shape[0], add_dummy_tokens), x.item(), dtype=x.dtype, device=x.device)], dim=1)
                if input_pos.numel() == 1:
                    start = input_pos.item()
                    input_pos_1 = torch.tensor([list(range(start, start + add_dummy_tokens + 1))], dtype=torch.long, device=x.device)
                else:
                    raise ValueError("input_pos should be a single value, but got {}".format(input_pos.shape))
                outs = self.model_dict["model"](x_1, input_pos=input_pos_1, **model_intput_kwargs)
        logits = outs["logits"]         # [bs, seqlen, vocab_size]
        logits_dim = logits.shape[-1]
        # Select the predicted token
        if add_dummy_tokens == 0 and logits.shape[1] > 1:     # Prefill step
            logits = logits.gather(index=real_pos.unsqueeze(-1).expand(-1, -1, logits_dim), dim=1).squeeze(1)
        elif add_dummy_tokens > 0:
            logits = logits[:, 0, :]
        else:   # Decode steps
            logits = logits[:, -1, :]   # [bs, vocab_size]
        if return_logits:
            out_logits = logits.clone()
        if penalty != 1.0:
            if x.shape[1] == 1:
                input_ids = torch.cat((next_token_ids, x), dim=1)
            else:
                input_ids = x
            score = torch.gather(logits, 1, input_ids)
            score = torch.where(score < 0, score * penalty, score / penalty)
            logits = score.scatter(1, input_ids, score)
        if logits_processor is not None:
            logits = logits_processor(x, logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)     # [bs, vocab_size]

        if logits_processor is not None:
            slice_vocab = logits_processor.get_processor("SliceVocabLogitsWarper")
            if slice_vocab is not None:
                assert not return_logits, "SliceVocabLogitsWarper should not be used with return_logits=True"
        else:
            slice_vocab = None

        next_token_list = []
        for batch_idx, generator in enumerate(generators):
            if do_sample:
                next_token = torch.multinomial(probs[batch_idx], num_samples=1, generator=generator)
            else:
                next_token = torch.argmax(probs[batch_idx], dim=-1, keepdim=True)
            # If SliceVocab in logits processor, we need to add the slice_start back
            if slice_vocab is not None:
                ind = next_token[0].cpu().item()
                if ind < slice_vocab.vocab_end - slice_vocab.vocab_start:
                    next_token += slice_vocab.vocab_start
                else:
                    cum_len = slice_vocab.vocab_end - slice_vocab.vocab_start
                    for other_slice in slice_vocab.other_slices:
                        if ind - cum_len < other_slice[1] - other_slice[0]:
                            next_token += other_slice[0] - cum_len
                        else:
                            cum_len += other_slice[1] - other_slice[0]
            next_token_list.append(next_token)
        next_token = torch.cat(next_token_list, dim=0).unsqueeze(-1)    # [bs, 1]

        if "past_key_values" in outs:
            next_token = {"next_token": next_token, "past_key_values": outs["past_key_values"]}
        if return_logits:
            return next_token, out_logits
        return next_token

    def _get_logits_processor(self, kwargs):
        if kwargs.get('do_sample', self.args.do_sample):
            logits_processor = kwargs.get('logits_processor', self.model_dict["logits_processor"])

            logits_processor_kwargs = {}
            if (top_k := kwargs.get('top_k')) is not None:
                logits_processor_kwargs['top_k'] = top_k
            if (top_p := kwargs.get('top_p')) is not None:
                logits_processor_kwargs['top_p'] = top_p
            if (temperature := kwargs.get('temperature')) is not None:
                logits_processor_kwargs['temperature'] = temperature

            if logits_processor_kwargs:
                logits_processor.update(**logits_processor_kwargs)

        else:
            logits_processor = None

        return logits_processor

    @torch.no_grad()
    def generate(
            self,
            prompts: Optional[Union[str, List[str]]] = None,
            pixel_values: Optional[torch.Tensor] = None,
            verbose: int = 1,
            **kwargs,
    ):
        """

        Parameters
        ----------
        prompts: Optional[Union[str, List[str]]]
            The input tokens for the model. Shape [batch_size, seq_len]
        pixel_values: Optional[torch.Tensor]
            The pixel values for the model. Shape [batch_size, c, h, w]. Values should be in [-1, 1] range.
        verbose: int
            The verbosity level. 0 for silent, 1 for detailed info.
        kwargs
        """
        out_dict = {}

        block_size = kwargs.get('block_size', self.args.block_size)
        seed = kwargs.get('seed')
        stop_id = kwargs.get('stop_id', self.tkwrapper.eos_token)
        return_first_pred_token_logits = kwargs.get('return_first_pred_token_logits', False)  # for mmlu_bench evaluation
        if return_first_pred_token_logits:
            first_pred_token_logits = None

        # Input sanity check
        if prompts is None and pixel_values is None:
            raise ValueError("Either prompts or pixel_values should be provided.")

        bsz, bsz2 = None, None
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            bsz = len(prompts)

        if pixel_values is not None:
            bsz2, _, ph, pw = pixel_values.shape
            th = ph // VISION_ENCODER_META_INFO[self.args.vision_model_type]["downsample_factor"][0]
            tw = pw // VISION_ENCODER_META_INFO[self.args.vision_model_type]["downsample_factor"][1]

        if bsz is not None and bsz2 is not None:
            assert bsz == bsz2, f"Batch size of prompts ({bsz}) and pixel_values ({bsz2}) should be the same."
        if bsz is None:
            bsz = bsz2

        seeds = self.prepare_seed(
            seed=seed,
            batch_size=bsz,
            num_sample_per_prompt=1,
        )
        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        logits_processor = self._get_logits_processor(kwargs)
        if verbose == 1:
            info_str = f"""
                  prompt: {prompts}
                    seed: {[gen.initial_seed() for gen in generators]}
        logits_processor: {logits_processor}
                    """
            self.logger.info(info_str)

        tokenizer: TokenizerWrapper = self.model_dict["tokenizer"]
        image_slices = [None]

        if pixel_values is None:
            inputs, real_pos = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_lm,
                prompt_list=prompts,
                infer_fn_kwargs_list=[dict(
                    add_eos=False,
                    return_text_mask=False,
                ) for _ in range(bsz)],
            )
            model_input_kwargs = dict()
        else:
            if prompts is None:
                prompts = [[] for _ in range(bsz)]
            inputs, iw_ih_scatter_index, image_slices, _, image_mask, real_pos = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion_mmu,
                prompt_list=prompts,
                infer_fn_kwargs_list=[dict(
                    image_token_lengths=th * tw,
                    max_token_length=None,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                    add_eos=False,
                    use_und_token=self.args.get('use_und_token', False),
                ) for _ in range(bsz)]
            )
            model_input_kwargs = dict(
                iw_ih_scatter_index=iw_ih_scatter_index.to(self.device),
                iw_ih_scatter_src=torch.tensor([[pw, ph]], dtype=torch.long, device=self.device),
                und_images=pixel_values.to(self.device),
                und_image_masks=image_mask.to(self.device),
            )

        inputs = inputs.to(self.device)

        # Get the real end position of the input for batch inference. Shape [bsz, 1]
        real_pos = real_pos.to(self.device)
        input_pos = torch.arange(0, inputs.shape[1], dtype=torch.long, device=self.device)

        self.model_dict["model"].set_kv_cache(batch_size=inputs.shape[0], device=self.device,
                                              image_slices=image_slices[0])
        infer_max_steps = block_size - real_pos.max()
        assert infer_max_steps > 0, f"The max number of generated tokens ({infer_max_steps}) should be greater than 0."

        start_time = time.time()
        next_token_ids = torch.empty([inputs.shape[0], 0], dtype=torch.long, device=self.device)
        stop_flag = torch.zeros([inputs.shape[0]], dtype=torch.bool, device=self.device)
        current_pos = real_pos
        pbar = range(infer_max_steps)
        if self.rank == 0:
            pbar = tqdm(pbar, desc="Generating text", leave=False)
        for i in pbar:
            if return_first_pred_token_logits and first_pred_token_logits is None:
                next_token, first_pred_token_logits = self.get_next_token(
                    inputs,
                    input_pos,
                    generators=generators,
                    real_pos=current_pos - 1,
                    logits_processor=logits_processor,
                    return_logits=True,
                    **(model_input_kwargs if i == 0 else {}),
                )
            else:
                next_token = self.get_next_token(
                    inputs,
                    input_pos,
                    generators=generators,
                    real_pos=current_pos - 1,
                    logits_processor=logits_processor,
                    **(model_input_kwargs if i == 0 else {}),
                )
            next_token_ids = torch.cat([next_token_ids, next_token], dim=-1)
            inputs = next_token
            input_pos = current_pos.clone()
            current_pos += 1
            # Update stop flag and determine whether to break
            stop_flag |= (next_token.squeeze(-1) == stop_id)
            if stop_flag.all():
                break
        # must clear kv cache in interactive mode
        self.model_dict["model"].clear_kv_cache()

        # Decode generated ids to text
        texts = []
        for ids in next_token_ids:
            stop_id_pos = torch.where(ids == stop_id)[0]
            if len(stop_id_pos) > 0:
                ids = ids[:stop_id_pos[0]]
            text = self.tkwrapper.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        gen_time = time.time() - start_time
        out_dict['samples'] = texts
        if return_first_pred_token_logits:
            out_dict["first_pred_token_logits"] = first_pred_token_logits if first_pred_token_logits is not None else [None] * len(texts)
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")

        return out_dict

    def batch_generate(self,
                       dataloader,
                       save_dir,
                       **kwargs):
        infer_kwargs = self.get_infer_kwargs()
        for key, value in infer_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        formatter = kwargs.get("formatter", lambda x: x)

        # Start sampling
        total_batches = len(dataloader)
        for batch_index, batch in enumerate(dataloader, start=1):
            batch: Dict[str, Any]
            self.logger.info(f"Batch {batch_index}/{total_batches}")

            inputs = {}
            if "type" in batch:
                inputs[batch["type"][0]] = batch["input"]
            for key in ["seed", "prompt", "pixel_values"]:
                if key in batch:
                    inputs[key] = batch[key]

            prompts = formatter(inputs.get("prompt"))
            pixel_values = inputs.get("pixel_values")
            assert prompts is not None or pixel_values is not None, "Either prompts or pixel_values should be provided."

            answers = self.generate(
                prompts=prompts,
                pixel_values=pixel_values,
                seed=inputs.get("seed"),
                **kwargs.get("generate_kwargs", {}),
            )["samples"]

            answers_df = [dict(
                index=batch["id"][i].item(),
                seed=inputs["seed"][i].item(),
                **(dict(question=inputs["prompt"][i]) if "prompt" in inputs else {}),
                answer=ans,
            ) for i, ans in enumerate(answers)]
            BaseSampler.save_batch_data(answers_df, save_dir / f"{self.rank}.csv")

    @torch.no_grad()
    def predict(self, prompt, **kwargs):
        out_dict = {}

        guidance_scale = kwargs.get("guidance_scale", self.args.guidance_scale)
        diff_infer_steps = kwargs.get("diff_infer_steps", self.args.diff_infer_steps)
        flow_shift = kwargs.get("flow_shift", self.args.sample_flow_shift)
        output_type = kwargs.get("output_type", "pil")
        verbose = kwargs.get("verbose", 1)
        uncond_enabled = kwargs.get("uncond_enabled")

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
        patch_size = self.args.patch_size

        assert h % (self.vae_downsample_factor[0] * patch_size) == 0 and w % (self.vae_downsample_factor[1] * patch_size) == 0, f"Image size should be divisible by vae_downsample_factor * patch_size, but got ({h} x {w}) with vae_downsample_factor={self.vae_downsample_factor} and patch_size={patch_size}"

        tk_height = h // (self.vae_downsample_factor[0] * patch_size)
        tk_width = w // (self.vae_downsample_factor[1] * patch_size)
        actual_image_token_length = tk_height * tk_width

        image_token_shape_wh = torch.tensor([w, h], dtype=torch.long).unsqueeze(0).to(self.device)
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
            do_classifier_free_guidance=self.args.guidance_scale > 1.0,
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

        # repeat for cfg
        if self.args.guidance_scale>1.0:
            iw_ih_scatter_src=iw_ih_scatter_src.repeat(2, 1)

        # with cond and uncond
        model_input_extra_kwargs = dict(
            idx=tokens,  # [b, 512]
            image_mask=image_mask,    # [b, 512]
            attention_mask=attention_mask,  # [b, 512, 512]
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

    def batch_sample(self,
                     dataloader,
                     input_batch_dict=None,
                     rerank=1,
                     **kwargs):
        infer_kwargs = self.get_infer_kwargs()
        for key, value in infer_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        formatter = kwargs.get("formatter", lambda x: x)
        skip_t2i = kwargs.get("skip_t2i", False)

        # Rerank Prepare
        rerank_enabled = False
        if rerank > 1:
            rerank_enabled = True
            self.load_clip_score_model()
        else:
            self.rerank_clip_score_metric = None

        # Start sampling
        total_batches = len(dataloader)
        for batch_index, batch in enumerate(dataloader, start=1):
            batch: Dict[str, Any]
            self.logger.info(f"Batch {batch_index}/{total_batches}")
            # Adjust max_width according to your terminal width
            self.logger.info(f"\n{batch_data_repr(batch, max_width=150)}")

            inputs = {}
            if "type" in batch:
                inputs[batch["type"][0]] = batch["input"]
            if "seed" in batch:
                inputs["seed"] = batch["seed"]
            for k, v in default(input_batch_dict, {}).items():
                inputs[k] = batch[v]

            prompts = formatter(inputs["prompt"])
            answers = self.generate(
                prompts=prompts,
                seed=inputs.get("seed"),
                **kwargs.get("generate_kwargs", {}),
            )["samples"]
            answers_df = pd.DataFrame([
                {
                    "index": batch["id"][i].item(),
                    "seed": inputs["seed"][i].item(),
                    "prompt": inputs["prompt"][i],
                    "recaption": ans,
                }
                for i, ans in enumerate(answers)
            ])
            inputs["prompt"] = answers

            if skip_t2i:
                save_names, final_samples = [], []
                prompts_save_path = Path(batch["save_path"][0]).parent / f"revised_prompts_{self.rank}.csv"
            else:
                if rerank_enabled:
                    outputs = self.rerank_wrapper(
                        fn=self.predict,
                        rerank=rerank,
                        input_name="prompt",
                        inputs=inputs['prompt'],
                        seeds=inputs['seed'],
                        output_type="pt",
                        **kwargs,
                    )
                else:
                    outputs = self.predict(
                        **inputs,
                        verbose=1,
                        **kwargs,
                    )
                final_samples = outputs["samples"]

                save_names = [
                    batch["save_path"][i // self.args.num_sample_per_prompt].format(
                        i % self.args.num_sample_per_prompt
                    )
                    for i, sample in enumerate(final_samples)
                ]
                prompts_save_path = Path(save_names[0]).parent / f"revised_prompts_{self.rank}.csv"

            # Save prompts when prompts is not None
            self.save_batch_image(final_samples, save_names, answers_df, prompts_save_path)


def x2image_interactive(input_fn, args, sampler, formatter, logger=None, **kwargs):
    """
    A wrapper for starting the interactive inference pipeline.
    """
    if logger is None:
        from loguru import logger
    while True:
        inputs = input_fn()
        if inputs is None:
            break

        # Determine the seed
        if args.seed_type in ["auto", "fixed"]:
            seed = args.seed
        elif args.seed_type == "random":
            seed = None
        else:
            raise ValueError(
                f"When evaluating `prompt`, `seed_type` must be one of ['auto', 'fixed', 'random'], "
                f"got {args.seed_type}."
            )

        revised_texts = sampler.generate(
            prompts=formatter(inputs["prompt"]),
            seed=seed,
        )['texts']
        inputs["prompt"] = revised_texts[0]

        # Start sampling
        outputs = sampler.predict(
            **inputs,
            size=args.sample_image_size,
            seed=seed,
            verbose=1,
            **kwargs,
        )
        samples = outputs["samples"]
        # Save the generated images
        if "prompt" in inputs:
            save_name = inputs["prompt"]
        elif "label" in inputs:
            save_name = str(inputs["label"])
        else:
            raise ValueError("Either `prompt` or `label` must be provided.")
        save_paths = sampler.get_default_sample_save_paths(len(samples), save_name, save_dir=args.sample_save_path)
        sampler.save_batch_image(samples, save_paths)
        logger.info(f"Save the generated image to: {save_paths}")


def recaption_then_text2image_interactive(args, sampler, logger=None, **kwargs):
    def input_fn():
        if args.prompt is None:
            # Ask for the next prompt
            inputs = input("Input prompt (`q` to quit): ")
            if inputs == "q":
                return None
            prompt = inputs
        else:
            prompt = args.prompt
            args.prompt = None
        return {'prompt': prompt}

    def formatter(prompt):
        return prompt + '<recaption>'

    x2image_interactive(input_fn, args, sampler, formatter, logger, **kwargs)


def recaption_then_text2image_batch(args, datasource, sampler, logger=None, segments=None, task=None,
                                    input_batch_dict=None, **kwargs):
    """
    A wrapper for starting the batch inference: recaption -> text2image.
    """
    if logger is None:
        from loguru import logger

    rerank = args.rerank if args.rerank > 1 else None
    save_base = sampler.get_sample_save_dir(
        task=task, testset=Path(args.csv).stem, image_size=args.sample_image_size, segments=segments, rerank=rerank,
    )
    logger.info(f"Save the generated results to: {save_base}")
    save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))

    def formatter(prompts):
        return [prompt + '<recaption>' for prompt in prompts]

    dataloader = sampler.build_sample_dataloader(
        data_source=datasource,
        save_template=save_template,
        batch_size=args.sample_batch_size,
        seed_type=args.seed_type,
        seed=args.seed,
        skip_exist=args.skip_exist,
    )
    sampler.batch_sample(
        dataloader=dataloader,
        input_batch_dict=input_batch_dict,
        rerank=args.rerank,
        # other kwargs passed to predict
        size=args.sample_image_size,
        formatter=formatter,
        **kwargs,
    )
    logger.info(f"Save the generated results to: {save_base}")


def captioning_batch(args, datasource, sampler, logger=None, segments=None, **kwargs):
    image_processor = sampler.model_dict["image_processor"]

    save_base = sampler.get_sample_save_dir(
        task="captioning", testset=Path(args.csv).stem, segments=segments, subdir="mmu",
    )
    logger.info(f"Save the generated answers to: {save_base}")

    dataset = ArrowDataset(
        arrow_file=datasource,
        length=None,
        save_template="{}",     # Not used.
        column_dict=dict(pixel_values="image@bytes"),
        seed=args.seed,
        seed_type=args.seed_type,
        logger=logger,
        image_processor=image_processor.preprocess,
    )
    dataloader = sampler.build_sample_dataloader(
        data_source=dataset,
        batch_size=args.sample_batch_size,
        skip_exist=args.skip_exist,
    )
    sampler.batch_generate(
        dataloader=dataloader,
        save_dir=save_base,
        **kwargs
    )
    logger.info(f"Save the generated answers to: {save_base}")


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)

    sampler = MultimodalTransfusionSampler.from_pretrained(
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
        if args.question:
            caller = lm_interactive
        elif args.image:
            caller = partial(image2x_interactive, processor="image_processor")
        elif args.task == "t2i":
            caller = text2image_interactive
        elif args.task == "recap_t2i":
            caller = recaption_then_text2image_interactive
        else:
            raise NotImplementedError(f"Unknown interactive task: {args.task}")

        caller(args, sampler, logger)

    elif args.task in ["captioning"]:
        segments = []
        infer_kwargs = sampler.get_infer_kwargs()
        top_p = infer_kwargs["top_p"]
        top_k = infer_kwargs["top_k"]
        temperature = infer_kwargs["temperature"]
        segments.append(f"t{temperature}_tp{top_p}_tk{top_k}")

        if args.task == "captioning":
            caller = captioning_batch
        else:
            raise ValueError(f"Unknown batch task: {args.task}")

        caller(
            args=args,
            datasource=args.csv,
            sampler=sampler,
            logger=logger,
            segments=segments,
        )

    elif args.task in ["sample", "recap", "recap_t2i", "recap_scan"]:
        segments = [
            f"{args.denoise_type}{args.diff_infer_steps}",
            f"cfg{args.guidance_scale}",
            f"recap",
        ]
        if args.task == "sample":
            caller = text2image_batch

        elif args.task in ["recap", "recap_t2i"]:
            infer_kwargs = sampler.get_infer_kwargs()
            top_p = infer_kwargs["top_p"]
            top_k = infer_kwargs["top_k"]
            temperature = infer_kwargs["temperature"]
            segments.append(f"t{temperature}_tp{top_p}_tk{top_k}")
            caller = partial(recaption_then_text2image_batch, skip_t2i=args.task == "recap")

        elif args.task == "recap_scan":
            caller = partial(recaption_then_text2image_batch, skip_t2i=True)
            # for temperature in [1.0, 0.9, 0.8, 0.7, 0.6]:
            #     for top_p in [0.95, 0.9, 0.8, 0.7, 0.6]:
            #         for top_k in [1024, 512, 256, 128, 64, 50]:
            for temperature in [0.6, 0.5]:
                for top_p in [0.65, 0.55, 0.4, 0.3]:
                    for top_k in [1024, 512, 256, 128, 64, 50]:
                        generate_kwargs = dict(top_p=top_p, top_k=top_k, temperature=temperature)
                        cur_segments = segments.copy()
                        cur_segments.append(f"t{temperature}_tp{top_p}_tk{top_k}")
                        caller(
                            args=args,
                            datasource=args.csv,
                            sampler=sampler,
                            logger=logger,
                            segments=cur_segments,
                            generate_kwargs=generate_kwargs,
                        )
        else:
            raise NotImplementedError(f"Unknown batch task: {args.task}")

        if "scan" not in args.task:
            caller(
                args=args,
                datasource=args.csv,
                sampler=sampler,
                logger=logger,
                segments=segments,
            )

    else:
        raise NotImplementedError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
