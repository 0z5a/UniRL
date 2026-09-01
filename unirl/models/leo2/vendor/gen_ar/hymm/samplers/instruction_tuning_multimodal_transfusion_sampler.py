import re
import time
import json
from functools import partial
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import easydict
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from hymm.config import parse_eval_initial_args
from hymm.constants import VISION_ENCODER_META_INFO
from hymm.data_kits.arrow_dataset import ArrowDataset
from hymm.data_kits.instruction_template import (
    text2image_instructions,
    inpainting_instructions,
    editing_instructions,
    subject_driven_instructions,
    face_id_instructions,
    grounding_mask_instructions,
    grounding_box_instructions,
)
from hymm.data_kits.instruction_tuning_transfusion_loader import InstructionTuningTransfusionArrowStream
from hymm.models.tokenizers.conversation import get_conversation_template
from hymm.samplers.base_sampler import setup_distributed_initialize, lm_interactive
from hymm.samplers.multimodal_transfusion_sampler import \
    MultimodalTransfusionSampler as MultimodalTransfusionSamplerBase
from hymm.utils.eval_utils import batch_data_repr
from hymm.utils.file_utils import rank0_logger
from hymm.utils.helpers import default, to_2tuple

MISSING_IMAGE_MSG = "Please provide the image."
MISSING_FACE_IMAGE_MSG = "We cannot detect the face in the image, please provide another image."
MISSING_FACE_ANALYSIS_MSG = "We are entering bof face identnty preserve mode, but the FaceAnalysis mode is not initialized."


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    else:
        return data


# When enabling --weights-only in pytorch>=2.4, we must manually allow the deserialization of EasyDict and set.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([set, easydict.EasyDict])


class MultimodalTransfusionSampler(MultimodalTransfusionSamplerBase):
    def apply_lm_template(self, prompt, max_length, system_prompt=None, think=False, answer=True):
        conv = get_conversation_template(self.conv_format)
        suffix = "<think>" if think else ("<answer>" if answer else "")
        system_prompt = f"{system_prompt}{conv.sep}" if system_prompt is not None else ""

        template = 'text'
        sections = [
            dict(type="text", text=f"{system_prompt}{conv.roles[0]}: {prompt}{conv.sep}{conv.roles[1]}: {suffix}"),
        ]
        output = self.tkwrapper.encode_general(
            template=template,
            sections=sections,
            max_token_length=max_length,
            use_text_mask=False,
            add_eos=False,
            add_pad=False,
        )
        tokens = output.tokens
        real_pos = output.real_pos
        return tokens, real_pos

    def apply_mmu_template(
            self,
            prompt_list,
            image_token_length,
            max_length,
            system_prompt=None,
            think=False,
            answer=True,
    ):
        bsz = len(prompt_list)

        conv = get_conversation_template(self.conv_format)
        suffix = "<think>" if think else ("<answer>" if answer else "")
        system_prompt = f"{system_prompt}{conv.sep}" if system_prompt is not None else ""

        text_max_length = max_length - image_token_length - 5   # 5 for (bos, und_boi, und_eoi, iw, ih)
        und_kwargs = dict(add_iw_ih_token=self.args.add_iw_ih_token, use_front_boi_token=self.args.use_front_boi_token)

        def _build_and_encode(prompt):
            template = 'text-und_image'
            sections = [
                dict(type="text", text=f"{system_prompt}{conv.roles[0]}: {prompt}{conv.sep}{conv.roles[1]}: {suffix}",
                     max_length=text_max_length),
                dict(type="und_image", token_length=image_token_length, **und_kwargs),
            ]
            return self.tkwrapper.encode_general(
                template=template,
                sections=sections,
                max_token_length=max_length,
                use_text_mask=False,
                add_eos=False,
                add_pad=False,
            )

        output = self.tkwrapper.batch_gen_infer(
            infer_fn=_build_and_encode,
            prompt_list=prompt_list,
        )

        n_tokens = output.tokens.shape[1]
        attention_mask = torch.ones(
            n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for image_slice in output.und_image_slices[i]:
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)

        return output, attention_mask

    def apply_t2i_template(
            self,
            prompt_list,
            sequence_template,
            text_max_length,
            image_token_length,
            image_size,
            cond_type=None,
            cond_token_lengths=None,
            answer=True,
            cfg_factor=1,
    ):
        assert cond_type in [None, "src_image", "face"], f"Invalid cond_type: {cond_type}"
        assert len(image_size) == 2, f"Invalid image size: {image_size}"
        th, tw = image_size
        bsz = len(prompt_list)
        if cond_token_lengths is None:
            cond_token_lengths = []
        n_conds = len(cond_token_lengths)
        
        # 5: <boi> <eoi> <timestep> (<iw> <ih> / <img_ratio>, <img_size>)
        if sequence_template == "instruct":
            # 11: <bos> 1, "User: " 3, "\n\n" 1, "Assistant: <answer>" 4, </answer> 1, <eos> 1
            max_length = text_max_length + image_token_length + sum(cond_token_lengths) + (n_conds + 1) * 5 + 11
            conv = get_conversation_template(self.conv_format)
            suffix = "<answer>" if answer else ""
        elif sequence_template == "pretrain":
            max_length = text_max_length + image_token_length + sum(cond_token_lengths) + (n_conds + 1) * 5

        gen_kwargs = dict(
            add_iw_ih_token=self.args.add_iw_ih_token, add_timestep_token=self.args.add_timestep_token,
            use_front_boi_token=self.args.use_front_boi_token,
        )

        def _build_and_encode(prompt, uncond_p=0.0):
            if sequence_template == "instruct":
                template = 'text-text-text' + f'-{cond_type}' * n_conds + '-image-text'
                sections = [
                    dict(type="text", text=f"{conv.roles[0]}: "),
                    dict(type="text", text=prompt, max_length=text_max_length,
                        uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p),
                    dict(type="text", text=f"{conv.sep}{conv.roles[1]}: " + suffix),
                ] + [
                    dict(type=cond_type, token_length=cond_token_length, **gen_kwargs)
                    for cond_token_length in cond_token_lengths
                ] + [
                    dict(type="gen_image", token_length=image_token_length, **gen_kwargs),
                    dict(type="text", text="</answer><|endoftext|>"),
                ]
            elif sequence_template == "pretrain":
                template = 'text' + f'-{cond_type}' * n_conds + '-image'
                sections = [
                    dict(type="text", text=prompt, max_length=text_max_length,
                        uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)
                ] + [
                    dict(type=cond_type, token_length=cond_token_length, **gen_kwargs)
                    for cond_token_length in cond_token_lengths
                ] + [
                    dict(type="gen_image", token_length=image_token_length, **gen_kwargs)
                ]
            return self.tkwrapper.encode_general(
                template=template,
                sections=sections,
                max_token_length=max_length,
                use_text_mask=False,
                add_eos=False,
                add_pad=False,
            )

        output = self.tkwrapper.batch_gen_infer(
            infer_fn=_build_and_encode,
            prompt_list=prompt_list,
            do_classifier_free_guidance=cfg_factor > 1,
            condition_repeat_times=1,
            uncondition_repeat_times=cfg_factor - 1,
        )

        n_images = 1 + (n_conds if cond_type == "src_image" else 0)
        iw_ih_src = torch.tensor([[tw, th] * n_images] * bsz * cfg_factor, dtype=torch.long)

        n_tokens = output.tokens.shape[1]
        attention_mask = torch.ones(
            n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0).repeat(bsz * cfg_factor, 1, 1)
        for i in range(bsz * cfg_factor):
            for image_slice in output.src_image_slices[i] + output.gen_image_slices[i]:
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)

        return output, iw_ih_src, attention_mask

    def x2image(
            self,
            prompt: Union[str, List[str]],
            generator: Union[torch.Generator, List[torch.Generator]],
            sequence_template: str = "instruct",
            pixel_values: Optional[Union[torch.Tensor, List[Union[torch.Tensor, List[torch.Tensor]]]]] = None,
            model_input_custom_kwargs: Optional[Dict[str, Any]] = None,
            cfg_factor: Optional[int] = None,
            save_paths: Optional[List[str]] = None,
            extra_log_info: str = "",
            pipeline_kwargs: Optional[Dict[str, Any]] = None,
            pbar=None,
            **kwargs,
    ):
        # `pixel_values` can be following types:
        # - None
        # - Tensor: [bsz, 3, H, W], range [-1, 1]
        # - [Tensor, [Tensor, Tensor]]: The list is bsz dim, and the first Tensor is [n_images, 3, H, W], the second
        #                               list has Tensors with shape [3, H, W] or [1, 3, H, W]. (H, W) can be different.
        # - [[Tensor], [Tensor, Tensor]]
        if pixel_values is not None:
            assert isinstance(pixel_values, (torch.Tensor, list)), \
                f"Invalid pixel_values type: {type(pixel_values)}. It should be a torch.Tensor or a list."
            if isinstance(pixel_values, list):
                assert all(len(pv) >= 1 for pv in pixel_values), "Each sample should have at least one image."

        args = self.args
        if isinstance(prompt, str):
            prompt = [prompt]
        if not isinstance(generator, list):
            generator = [generator]
        assert len(prompt) == len(generator), \
            f"The number of prompts and generators should be the same, but got {len(prompt)} and {len(generator)}."
        bsz = len(prompt)

        guidance_scale = kwargs.get("guidance_scale", args.guidance_scale)
        diff_infer_steps = kwargs.get("diff_infer_steps", args.diff_infer_steps)
        output_type = kwargs.get("output_type", "pil")
        verbose = kwargs.get("verbose", 1)
        ds_factor = self.vae_downsample_factor[0] * args.patch_size
        if cfg_factor is None:
            cfg_factor = 1 + (guidance_scale > 1.0)

        # -- target image size
        # There are a few situations:
        # 1. If no src images are provided (t2i), the target image size should be the input size.
        # 2. If one src image is provided (editing), the target image size should be the same as the src image.
        # 3. If multiple src images are provided (subject driven), the target image size should be the same as the
        #    last src image.
        if pixel_values is None:
            image_size = self.parse_image_size(kwargs.get("size", args.sample_image_size), align=16)
        elif isinstance(pixel_values, torch.Tensor):
            image_size = [pixel_values.shape[-2], pixel_values.shape[-1]]
        else:
            image_size = [pixel_values[0][-1].shape[-2], pixel_values[0][-1].shape[-1]]
        image_token_length = image_size[0] * image_size[1] // (ds_factor ** 2)

        #  -- flux shift
        scheduler_set_timesteps_extra_kwargs = {}
        if args.sample_use_flux_shift:
            scheduler_set_timesteps_extra_kwargs['n_tokens'] = image_token_length

        # -- vae encode
        if isinstance(pixel_values, torch.Tensor):
            # ti2i: batch src images have the same size, and each sample has one src image
            pixel_values = pixel_values.to(self.device)
            input_src_t, input_src_x = self.vae_encode(pixel_values, sample_type="sample_start", cfg_factor=cfg_factor)
            h_, w_ = input_src_x.shape[-2:]
            cond_type = "src_image"
            cond_token_lengths = [h_ * w_ // (args.patch_size ** 2)]

        elif isinstance(pixel_values, list):
            # ti2i: batch src images have different sizes, or each sample has different number of src images
            batch_src_t, batch_src_x, batch_src_image_token_lengths = [], [], []
            for item in pixel_values:
                if isinstance(item, torch.Tensor) and item.dim() == 3:
                    item = [item]
                src_xs, src_model_ts, src_tk_lengths = [], [], []
                for src_image in item:
                    if src_image.ndim != 3:
                        assert src_image.ndim == 4 and src_image.shape[0] == 1
                        src_image = src_image.squeeze(0)
                    src_image = src_image[None].to(self.device)
                    src_model_t, src_x = self.vae_encode(src_image, sample_type="sample_start")
                    src_xs.append(src_x[0])
                    src_model_ts.append(src_model_t)
                    h_, w_ = src_x.shape[-2:]
                    src_tk_lengths.append(h_ * w_ // (args.patch_size ** 2))
                if all([src_x.shape == src_xs[0].shape for src_x in src_xs]):
                    batch_src_x.append(torch.stack(src_xs))
                else:
                    batch_src_x.append(src_xs)
                batch_src_t.append(torch.cat(src_model_ts))
                batch_src_image_token_lengths.append(src_tk_lengths)
            input_src_t = batch_src_t * cfg_factor
            input_src_x = batch_src_x * cfg_factor
            cond_type = "src_image"
            cond_token_lengths = batch_src_image_token_lengths[0]

        else:
            # t2i
            assert pixel_values is None, f"Invalid pixel_values type: {type(pixel_values)}."
            input_src_t, input_src_x = None, None
            cond_type, cond_token_lengths = None, None

        # -- face id
        if "src_face_embedding" in default(model_input_custom_kwargs, {}):
            cond_type = "face"
            cond_token_lengths = [args.get('face_id_resampler_token_length', args.get('face_id_preserve_resampler_token_length'))]

        # Input one sample and output batched one
        output, iw_ih_src, attention_mask = self.apply_t2i_template(
            prompt, sequence_template, args.text_token_length, image_token_length, image_size,
            cond_type=cond_type, cond_token_lengths=cond_token_lengths, answer=True, cfg_factor=cfg_factor,
        )

        model_input_extra_kwargs = dict(
            idx=output.tokens.to(self.device),
            src_x=to_device(input_src_x, self.device),
            src_t=to_device(input_src_t, self.device),
            src_image_mask=to_device(output.src_image_mask, self.device),
            image_mask=output.gen_image_mask.to(self.device),
            attention_mask=attention_mask.to(self.device),
            iw_ih_scatter_index=output.iw_ih_scatter_index.to(self.device) if output.iw_ih_scatter_index is not None else None,
            iw_ih_scatter_src=iw_ih_src.to(self.device) if output.iw_ih_scatter_index is not None else None,
            timestep_scatter_index=output.timestep_scatter_index.to(self.device),
            und_image_masks=to_device(output.face_image_mask, self.device),
            **default(model_input_custom_kwargs, {}),
        )

        if verbose == 1:
            context = self.tkwrapper.tokenizer.decode(output.tokens[0], skip_special_tokens=False)
            # Replace <img><img>...<img> with [<img>]{number}
            context = re.sub(r"(<img>)+", lambda m: f"[<img>]{{{len(m.group(0)) // 5}}}", context)
            if '<face>' in context:
                context = re.sub(r"(<face>)+", lambda m: f"[<face>]{{{len(m.group(0)) // 6}}}", context)
            info_str = f"""
         token shape: {output.tokens.shape}
          context[0]: {context}
          image_size: {image_size}
                seed: {[g.initial_seed() for g in generator]}
         infer_steps: {diff_infer_steps}
      guidance_scale: {guidance_scale}
          flow_shift: {self.args.sample_flow_shift}{extra_log_info}"""
            self.logger.info(info_str)

        samples = self.pipeline(batch_size=bsz,
                                image_size=image_size,
                                num_inference_steps=diff_infer_steps,
                                guidance_scale=guidance_scale,
                                generator=generator,
                                output_type=output_type,
                                model_input_extra_kwargs=model_input_extra_kwargs,
                                scheduler_set_timesteps_extra_kwargs=scheduler_set_timesteps_extra_kwargs,
                                pbar=pbar,
                                pbar_steps=image_token_length,
                                **default(pipeline_kwargs, {}),
                                )[0]
        if save_paths is not None:
            self.save_batch_image(samples, save_paths)
        return samples

    def ask_input(self, input_type: str):
        while True:
            if input_type == "image":
                inputs = input("[Input image path (`q` to quit):] ")
                if inputs == "q":
                    inputs = None
                    break
                inputs_list = inputs.split(',')
                success = True
                for image_path in inputs_list:
                    if not Path(image_path).exists() and not Path(image_path).is_file():
                        print(f"File not found: {image_path}")
                        success = False
                        break
                if not success:
                    continue
                break

            else:
                raise NotImplementedError(f"Unknown input type: {input_type}")
        return str(inputs) if inputs is not None else None

    def _get_und_image_size(self, pixel_values):
        assert isinstance(pixel_values, torch.Tensor), "pixel_values should be a torch.Tensor."
        assert pixel_values.ndim == 4, "pixel_values should have 4 dimensions."
        assert pixel_values.shape[1] == 3, "pixel_values should have 3 channels."
        _, _, ph, pw = pixel_values.shape
        th = ph // VISION_ENCODER_META_INFO[self.args.vision_model_type]["downsample_factor"][0]
        tw = pw // VISION_ENCODER_META_INFO[self.args.vision_model_type]["downsample_factor"][1]
        return ph, pw, th * tw

    @torch.no_grad()
    def _generate(
            self,
            prompts: str,
            system_prompt: Optional[str] = None,
            pixel_values: Optional[List[Image.Image]] = None,
            think: bool = False,
            verbose: int = 1,
            skip_special_tokens=True,
            save_image: bool = False,
            retrieve_image_num: int = 1,
            ask_for_image: bool = False,
            **kwargs,
    ):
        """
        A uniform interface for all the t2i, ti2i\face, face, lm, and mmu tasks.
        The prompts should be the formatted string like:

        Parameters
        ----------
        prompts: str
            The input prompt for the model.
        system_prompt: Optional[str]
            The system prompt for the model. A string.
        pixel_values: Optional[List[Image.Image]]
            A list of PIL images for the model.
        think: bool
            Whether to add the <think> token to the prompt.
        verbose: int
            The verbosity level. 0 for silent, 1 for detailed info.
        skip_special_tokens: bool
            Whether to skip the special tokens.
        save_image: bool
            Whether to save the image.
        retrieve_image_num: int
            The number of images to retrieve from the end of the pixel_values list.
        ask_for_image: bool
            Whether to ask for the image if the pixel_values is empty. If False, when the pixel_values is empty,
            it will return a default message.
        kwargs
        """
        args = self.args
        if pixel_values is None:
            pixel_values = []
        else:
            assert isinstance(pixel_values, list), "pixel_values should be a list."

        # Prompt sanity check
        if isinstance(prompts, list):
            prompts = prompts[0]
        assert isinstance(prompts, str), f"`prompt` should be a string, but got {type(prompts)}."
        bsz = 1

        # Common arguments
        block_size = kwargs.get('block_size', args.block_size)
        seed = kwargs.get('seed')
        logits_processor = self._get_logits_processor(kwargs)
        #   -- seed
        seed = self.prepare_seed(
            seed=seed,
            batch_size=bsz,
            num_sample_per_prompt=1,
        )[0]
        generator = torch.Generator(self.device).manual_seed(seed)

        # LM, MMU
        stop_id = kwargs.get('stop_id', self.tkwrapper.eos_token)
        boi_id = kwargs.get('boi_id', self.tkwrapper.special_token_map['<boi>'])
        src_boi_id = kwargs.get('src_boi_id', self.tkwrapper.special_token_map['<src_boi>'])
        und_boi_id = kwargs.get('und_boi_id', self.tkwrapper.special_token_map['<und_boi>'])
        bof_id = kwargs.get('bof_id', self.tkwrapper.special_token_map['<bof>'])
        stream = kwargs.get("stream", False)

        if verbose == 1:
            info_str = f"""
                  prompt: {prompts}
                    seed: {generator.initial_seed()}
        logits_processor: {logits_processor}
                    """
            self.logger.info(info_str)
        # Apply the template to the prompts
        if system_prompt is not None:
            assert isinstance(system_prompt, str), f"`system_prompt` should be a string, but got {type(system_prompt)}."
        inputs, real_pos = self.apply_lm_template(
            prompts, block_size, system_prompt=system_prompt, think=think, answer=True)
        inputs = inputs[None].to(self.device)
        real_pos = real_pos[None].to(self.device)

        input_pos = torch.arange(0, inputs.shape[1], dtype=torch.long, device=self.device)
        self.model_dict["model"].set_kv_cache(batch_size=bsz, device=self.device)
        infer_max_steps = block_size - real_pos.max()

        next_token_ids = torch.empty([1, 0], dtype=torch.long, device=self.device)
        current_pos = real_pos
        model_input_kwargs = {}
        pbar = range(infer_max_steps)

        if self.rank == 0:
            pbar = tqdm(pbar, desc="Text mode", leave=False, disable=stream)
        for i in pbar:
            next_token = self.get_next_token(
                inputs,
                input_pos,
                generators=[generator],
                # `real_pos` only used in prefill stage. Here we use `real_pos - 1` instead of `current_pos - 1` to
                # allow encode und image tokens (using prefill) in the middle steps. In this way, one should modify
                # real_pos to the length of und image tokens.
                real_pos=real_pos - 1,
                logits_processor=logits_processor,
                **model_input_kwargs,
            )
            next_token_ids = torch.cat([next_token_ids, next_token], dim=1)
            inputs = next_token
            input_pos = current_pos.clone()
            current_pos += 1
            model_input_kwargs.clear()

            # Route
            next_token = next_token.item()
            if next_token == stop_id:
                break

            if next_token == boi_id:
                text = self.tkwrapper.tokenizer.decode(next_token_ids[0], skip_special_tokens=skip_special_tokens)
                if text:
                    yield {'role': 'Assistant', 'value': text, 'type': 'text'}

                pbar.set_description("Text to Image mode")
                if save_image:
                    save_name = prompts[:200]
                    save_paths = self.get_default_sample_save_paths(bsz, save_name, save_dir=self.args.sample_save_path)
                else:
                    save_paths = None
                samples = self.x2image(prompts, generator=generator, save_paths=save_paths, pbar=pbar, **kwargs)
                yield {'role': 'Assistant', 'value': samples, 'type': 'image', 'save_paths': save_paths}
                break

            elif next_token == src_boi_id:
                text = self.tkwrapper.tokenizer.decode(next_token_ids[0], skip_special_tokens=skip_special_tokens)
                if text:
                    yield {'role': 'Assistant', 'value': text, 'type': 'text'}

                pbar.set_description("Text+Image to Image mode")
                if len(pixel_values) == 0:
                    if ask_for_image:
                        image_paths = self.ask_input("image")
                        if image_paths is None:
                            break
                        image_paths = image_paths.split(',')
                        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
                    else:
                        yield {'role': 'Assistant', 'value': f"[{MISSING_IMAGE_MSG}]", 'type': 'text'}
                        break
                else:
                    images = pixel_values[-retrieve_image_num:]
                processed_images = self.model_dict['vae_processor'].preprocess(images)['pixel_values']
                if len(images) > 1:     # subject driven
                    processed_images = [processed_images]

                if save_image:
                    save_name = prompts[:200]
                    save_paths = self.get_default_sample_save_paths(bsz, save_name, save_dir=self.args.sample_save_path)
                else:
                    save_paths = None
                samples = self.x2image(prompts, generator=generator,
                                       pixel_values=processed_images, save_paths=save_paths, pbar=pbar, **kwargs)
                yield {'role': 'Assistant', 'value': samples, 'type': 'image', 'save_paths': save_paths}
                if not skip_special_tokens:
                    yield {'role': 'Assistant', 'value': '<src_eoi><|endoftext|>', 'type': 'text'}
                break

            elif next_token == und_boi_id:
                text = self.tkwrapper.tokenizer.decode(next_token_ids[0], skip_special_tokens=False)
                if text:
                    yield {'role': 'Assistant', 'value': text, 'type': 'text'}

                pbar.set_description("Image Understanding mode")
                if len(pixel_values) == 0:
                    if ask_for_image:
                        image_paths = self.ask_input("image")
                        if image_paths is None:
                            break
                        image_paths = image_paths.split(',')[-1]    # only use the last image if multiple images
                        images = [Image.open(image_paths).convert("RGB")]
                    else:
                        yield {'role': 'Assistant', 'value': f"[{MISSING_IMAGE_MSG}]", 'type': 'text'}
                        break
                else:
                    images = pixel_values[-1:]
                processed_images = self.model_dict['image_processor'].preprocess(images).pixel_values

                ph, pw, token_length = self._get_und_image_size(processed_images)

                template = 'und_image'
                sections = [
                    dict(type="und_image", token_length=token_length,
                         add_iw_ih_token=self.args.add_iw_ih_token, use_front_boi_token=self.args.use_front_boi_token),
                ]
                output = self.tkwrapper.encode_general(
                    template=template,
                    sections=sections,
                    max_token_length=None,
                    use_text_mask=False,
                    add_eos=False,
                    add_pad=False,
                    add_bos=False,
                )
                inputs = output.tokens[None].to(self.device)    # remove the <bos> token
                current_pos -= 1
                prefix = current_pos.item()
                input_pos = torch.arange(0, inputs.shape[1], dtype=torch.long, device=self.device)[None] + current_pos
                current_pos += inputs.shape[1]
                real_pos = output.real_pos[None].to(self.device)
                model_input_kwargs.update(dict(
                    iw_ih_scatter_index=output.iw_ih_scatter_index[None].to(self.device),
                    iw_ih_scatter_src=torch.tensor([[pw, ph]], dtype=torch.long, device=self.device),
                    und_images=processed_images.to(self.device),
                    und_image_masks=output.und_image_mask[None].to(self.device),
                ))

                image_slices = [slice(sli.start + prefix, sli.stop + prefix) for sli in output.und_image_slices]
                self.model_dict["model"].update_mask_cache(image_slices)

                # Reset next_token_ids to prepare new responses
                next_token_ids = torch.empty([1, 0], dtype=torch.long, device=self.device)

            elif next_token == bof_id:
                text = self.tkwrapper.tokenizer.decode(next_token_ids[0], skip_special_tokens=skip_special_tokens)
                if text:
                    yield {'role': 'Assistant', 'value': text, 'type': 'text'}
                if "face_analysis" not in self.model_dict:
                    yield {'role': 'Assistant', 'value': f"[{MISSING_FACE_ANALYSIS_MSG}]", 'type': 'text'}
                    break
                pbar.set_description("FaceID mode")
                # face crop and face emebedding encode logic
                images = pixel_values[-1]
                processed_images = self.model_dict['face_image_processor'].preprocess(images)
                image_tensor_norm, face_embedding, face_bbox = InstructionTuningTransfusionArrowStream.crop_face_image_torch(
                    self.model_dict["face_analysis"], 
                    processed_images, 
                    max_side=(256, 256,), 
                    logger=self.logger, 
                    src_condition_type="face_id",
                    return_bbox=True
                )
                if face_embedding is None:
                    yield {'role': 'Assistant', 'value': f"[{MISSING_FACE_IMAGE_MSG}]", 'type': 'text'}
                    break
                # face cfg is enabeld by default
                face_guidance_scale = kwargs.get("face_guidance_scale", self.args.face_guidance_scale)
                guidance_scale = kwargs.get("guidance_scale", 1.0)
                cfg_factor = 1 + (guidance_scale > 1.0) + (face_guidance_scale > 1.0)
                src_face_embedding = face_embedding.unsqueeze(-1).unsqueeze(-1).to(self.device)
                src_face_embedding = src_face_embedding.repeat(cfg_factor, 1, 1, 1)
                face_uncond_chunk_index = (2 if guidance_scale > 1.0 else 1) if face_guidance_scale > 1.0 else None
                src_face_embedding[face_uncond_chunk_index, ...] = torch.zeros_like(
                        src_face_embedding[face_uncond_chunk_index, ...])
                model_input_custom_kwargs = dict(
                    src_face_embedding=src_face_embedding,
                )
                pipeline_kwargs = dict(
                    face_guidance_scale=face_guidance_scale,
                )
                if save_image:
                    save_name = prompts[:200]
                    save_paths = self.get_default_sample_save_paths(bsz, save_name, save_dir=self.args.sample_save_path)
                else:
                    save_paths = None
                samples = self.x2image(
                    prompts, 
                    generator=generator, 
                    pixel_values=None,      # pixel value will be none for face id generation
                    cfg_factor=cfg_factor,
                    save_paths=save_paths, 
                    pbar=pbar, 
                    model_input_custom_kwargs=model_input_custom_kwargs,
                    pipeline_kwargs=pipeline_kwargs,
                    **kwargs
                )
                yield {'role': 'Assistant', 'value': samples, 'type': 'image', 'save_paths': save_paths}
                if not skip_special_tokens:
                    yield {'role': 'Assistant', 'value': '<src_eoi><|endoftext|>', 'type': 'text'}
                break
            else:
                pbar.set_description("Text mode")

            text = self.tkwrapper.tokenizer.decode(next_token, skip_special_tokens=skip_special_tokens)
            yield {'role': 'Assistant', 'value': text, 'type': 'text'}

        # must clear kv cache in interactive mode
        # out_dict["text"] = text
        self.model_dict["model"].clear_kv_cache()

    def generate(self, *args, **kwargs):
        stream = kwargs.get("stream", False)
        if stream:
            return self._generate(*args, **kwargs)
        else:
            return list(self._generate(*args, **kwargs))

    @torch.no_grad()
    def batch_t2i(self, prompt, task=None, src_img_tensor_batch=None, sequence_template="instruct", **kwargs):
        out_dict = {}
        verbose = kwargs.get("verbose", 1)
        assert task, "Task should be provided."

        # -- Prompt and seeds
        condition_dict, bsz = self.prepare_prompts(prompt, **kwargs)
        raw_prompt = condition_dict['prompt']

        seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=bsz, num_sample_per_prompt=1)
        out_dict["seeds"] = seeds
        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        assert sequence_template in ["pretrain", "instruct"], "sequence_template should be either 'pretrain' or 'instruct'."
        if sequence_template == "instruct":
            task_instructions = dict(
                t2i=text2image_instructions,
                inpainting=inpainting_instructions,
                editing=editing_instructions,
                subject_driven=subject_driven_instructions,
                face_id=face_id_instructions,
                grounding=grounding_mask_instructions,
            )
            user_prompt = []
            for p, seed in zip(raw_prompt, seeds):
                instruct = task_instructions[task][seed % len(task_instructions[task])]
                user_prompt.append(f"{instruct} {p}")
        elif sequence_template == "pretrain":
            print("here")
            user_prompt = raw_prompt

        out_dict['prompt'] = user_prompt
        # -- Source images

        if task == "face_id":
            # Process face embedding
            assert self.args.face_guidance_scale is not None, "face_guidance_scale should be provided for face_id task."
            assert "src_face_embedding" in kwargs, "`src_face_embedding` should be provided for face_id task."
            src_face_embedding = kwargs["src_face_embedding"]
            assert isinstance(src_face_embedding, torch.Tensor), "src_face_embedding should be a torch.Tensor."
            src_face_embedding = src_face_embedding.unsqueeze(-1).unsqueeze(-1).to(self.device)     # [b, c, 1, 1]

            guidance_scale = kwargs.get("guidance_scale", self.args.guidance_scale)
            face_guidance_scale = kwargs.get("face_guidance_scale", self.args.face_guidance_scale)
            cfg_factor = 1 + (guidance_scale > 1.0) + (face_guidance_scale > 1.0)
            if cfg_factor > 1:
                src_face_embedding = src_face_embedding.repeat(cfg_factor, 1, 1, 1)
                face_uncond_chunk_index = (2 if guidance_scale > 1.0 else 1) if face_guidance_scale > 1.0 else None
                if face_uncond_chunk_index is not None:
                    face_uncond_chunk_range = slice(face_uncond_chunk_index * bsz, (face_uncond_chunk_index + 1) * bsz)
                    src_face_embedding[face_uncond_chunk_range, ...] = torch.zeros_like(
                        src_face_embedding[face_uncond_chunk_range, ...])

            extra_log_info = f"""
 face_guidance_scale: {face_guidance_scale}"""

            kwargs.update(dict(
                model_input_custom_kwargs=dict(
                    src_face_embedding=src_face_embedding,
                ),
                cfg_factor=cfg_factor,
                extra_log_info=extra_log_info,
                pipeline_kwargs=dict(
                    face_guidance_scale=face_guidance_scale,
                )
            ))

        # -- Batch Run
        start_time = time.time()
        out_dict["samples"] = self.x2image(user_prompt, generator=generators, pixel_values=src_img_tensor_batch, sequence_template=sequence_template, **kwargs)
        gen_time = time.time() - start_time
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")
        return out_dict

    @torch.no_grad()
    def batch_mmu(self, prompt, image: torch.Tensor, **kwargs):
        """ The image has shape of [bsz, 3, H, W] and ranges in [-1, 1] """
        args = self.args
        out_dict = {}
        verbose = kwargs.get("verbose", 1)
        block_size = kwargs.get('block_size', args.block_size)
        logits_processor = self._get_logits_processor(kwargs)

        # -- Prompt and seeds
        if isinstance(prompt, str):
            prompt = [prompt]
        out_dict['prompt'] = prompt
        bsz = len(prompt)

        seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=bsz, num_sample_per_prompt=1)
        out_dict["seeds"] = seeds
        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        # -- Understanding images
        ph, pw, image_token_length = self._get_und_image_size(image)
        iw_ih_src = torch.tensor([[pw, ph]] * bsz, dtype=torch.long, device=self.device)

        output, attention_mask = self.apply_mmu_template(
            prompt, image_token_length, max_length=block_size, answer=True,
        )
        inputs = output.tokens.to(self.device)
        real_pos = output.real_pos.to(self.device)
        assert real_pos.ndim == 2, f"Invalid real_pos shape: {real_pos.shape}"

        if verbose == 1:
            context = self.tkwrapper.tokenizer.decode(output.tokens[0], skip_special_tokens=False)
            # Replace <img><img>...<img> with [<img>]{number}
            context = re.sub(r"(<img>)+", lambda m: f"[<img>]{{{len(m.group(0)) // 5}}}", context)
            info_str = f"""
             token shape: {output.tokens.shape}
              context[0]: {context}
                    seed: {[g.initial_seed() for g in generators]}
        logits_processor: {logits_processor}
                    """
            self.logger.info(info_str)

        model_input_kwargs = dict(
            iw_ih_scatter_index=output.iw_ih_scatter_index.to(self.device),
            iw_ih_scatter_src=iw_ih_src,
            und_images=image.to(self.device),
            und_image_masks=output.und_image_mask.to(self.device),
        )

        start_time = time.time()
        batch_input_pos = torch.arange(
            0, inputs.shape[1], dtype=torch.long, device=self.device)[None].expand(
            bsz, -1)     # use expand to share indices to save memory
        self.model_dict["model"].set_kv_cache(batch_size=bsz, device=self.device, image_slices=output.und_image_slices)
        infer_max_steps = block_size - real_pos.max()

        next_token_ids = torch.empty([bsz, 0], dtype=torch.long, device=self.device)
        stop_flag = torch.zeros([bsz], dtype=torch.bool, device=self.device)
        current_pos = real_pos
        pbar = range(infer_max_steps)
        if self.rank == 0:
            pbar = tqdm(pbar, desc="Text mode", leave=False)
        for i in pbar:
            next_token = self.get_next_token(
                inputs,
                batch_input_pos,
                generators=generators,
                real_pos=real_pos - 1,
                logits_processor=logits_processor,
                **(model_input_kwargs if i == 0 else {}),
            )
            next_token_ids = torch.cat([next_token_ids, next_token], dim=1)
            inputs = next_token
            batch_input_pos = current_pos.clone()
            current_pos += 1
            # Update stop flag and determine whether to break
            stop_flag |= (next_token.squeeze(-1) == self.tkwrapper.eos_token)
            if stop_flag.all():
                break

        # Clear kv cache
        self.model_dict["model"].clear_kv_cache()

        # Decode generated ids to text
        texts = []
        for ids in next_token_ids:
            stop_id_pos = torch.where(ids == self.tkwrapper.eos_token)[0]
            if len(stop_id_pos) > 0:
                ids = ids[:stop_id_pos[0]]
            text = self.tkwrapper.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        out_dict['samples'] = texts
        gen_time = time.time() - start_time
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")
        return out_dict

    def batch_sample_general(
            self,
            dataloader,
            run_fn,
            input_batch_dict=None,
            run_fn_kwargs=None,
            save_dir=None,
            rerank=1,
            **kwargs,
    ):
        infer_kwargs = self.get_infer_kwargs()
        for key, value in infer_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

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

            run_fn_kwargs = default(run_fn_kwargs, {})
            if rerank_enabled:
                outputs = self.rerank_wrapper(
                    fn=run_fn, rerank=rerank, input_name="prompt", **inputs, output_type="pt", **run_fn_kwargs,
                )
            else:
                outputs = run_fn(**inputs, verbose=1, **run_fn_kwargs)
            final_samples = outputs["samples"]

            # Save all types of results
            if isinstance(final_samples[0], str):
                assert isinstance(save_dir, Path), "save_dir should be a Path object."
                answers = [dict(
                    index=batch["id"][i].item(),
                    answer=ans,
                    **(dict(seed=inputs["seed"][i].item()) if "seed" in inputs else {}),
                    **(dict(question=inputs["prompt"][i]) if "prompt" in inputs else {}),
                ) for i, ans in enumerate(final_samples)]
                self.save_batch_data(answers, save_dir / f"{self.rank}.csv")

            elif isinstance(final_samples[0], Image.Image):
                save_names = [
                    batch["save_path"][i // self.args.num_sample_per_prompt].format(
                        i % self.args.num_sample_per_prompt
                    )
                    for i, sample in enumerate(final_samples)
                ]
                # Save prompts when prompts is not None
                self.save_batch_image(final_samples, save_names)

            else:
                raise ValueError(f"Invalid final_samples type: {type(final_samples[0])}")


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
        # `ask_for_image` means when enter the src_boi/und_boi mode, it will ask use for the image path.
        # If set to False, it will directly return a default message.
        lm_interactive(args, sampler, logger, stream=args.stream, save_image=True, ask_for_image=True)
        return

    # Batch sampling

    if args.task in ["t2i", "inpainting", "editing", "subject_driven", "face_id", "grounding"]:
        segments = [f"{args.denoise_type}{args.diff_infer_steps}", f"cfg{args.guidance_scale}"]

        dataset_kwargs = None

        # column_dict: {batch key <- data column}                   used in dataset
        # input_batch_dict: {predict fn input key <- batch key}     used in batch_ti2i

        if args.task == "t2i":
            dataset = args.csv
            input_batch_dict = {}

        elif args.task == "inpainting":
            dataset = args.csv
            def get_item_callback(item_dict):
                # 3 x h x w
                src_image_tensor = sampler.vae_processor.normalize(Image.open(item_dict["extra_src_img_path"]))
                # 1 x h x w
                mask_image_tensor = transforms.ToTensor()(Image.open(item_dict["extra_mask_img_path"]))
                masked_img_tensor = src_image_tensor * (1 - (mask_image_tensor > 0.5).float())
                del item_dict["extra_src_img_path"]
                del item_dict["extra_mask_img_path"]
                item_dict["src_img_tensor_batch"] = masked_img_tensor

                item_dict["caption"] = json.loads(item_dict["extra_caption"])["long_caption"]
                del item_dict["extra_caption"]
                return item_dict

            extra_cols = ["caption", "src_img_path", "mask_img_path"]
            dataset_kwargs = {"extra_cols": extra_cols, "callback": lambda x: x, "get_item_callback": get_item_callback}
            input_batch_dict = dict(
                prompt="caption",
                src_img_tensor_batch="src_img_tensor_batch",
            )

        elif args.task in ["editing", "grounding"]:
            dataset = args.csv
            def get_item_callback(item_dict):
                # 3 x h x w
                pil_image = Image.open(item_dict["extra_src_img_path"]).resize((512, 512))
                src_image_tensor = sampler.vae_processor.normalize(pil_image)
                del item_dict["extra_src_img_path"]
                item_dict["src_img_tensor_batch"] = src_image_tensor

                return item_dict

            extra_cols = ["src_img_path"]
            dataset_kwargs = {"extra_cols": extra_cols, "callback": lambda x: x, "get_item_callback": get_item_callback}
            input_batch_dict = dict(
                prompt="input",
                src_img_tensor_batch="src_img_tensor_batch"
            )

        elif args.task == "subject_driven":
            dataset = args.csv

            if "dreambooth" in args.csv:
                ref_cols = ['ref1']
            elif "subject_driven_two_subject_ImageHub" in args.csv:
                ref_cols = ['ref1', 'ref2']
            else:
                ref_cols = ['ref1', 'ref2', 'ref3']

            def get_item_callback(item_dict):
                src_img_tensor_list = []
                for obj in ref_cols:
                    if Path(item_dict[f'extra_{obj}']).exists():
                        # follow ArrowDataset, we only apply ToTensor() transform
                        src_img_tensor_list.append(
                            sampler.vae_processor.normalize(Image.open(item_dict[f'extra_{obj}']).convert('RGB'))
                        )
                    del item_dict[f'extra_{obj}']
                item_dict['src_img_tensor_batch'] = [src_img_tensor_list]  # after default collate_fn, become 1 * [n_srcs * [ 1 x c x h x w ]]
                return item_dict

            dataset_kwargs = {'extra_cols': ref_cols, 'get_item_callback': get_item_callback}
            input_batch_dict = dict(
                prompt="input",
                src_img_tensor_batch="src_img_tensor_batch",
            )

        elif args.task == "face_id":
            dataset = args.csv

            def get_item_callback(item_dict):
                # 512
                src_face_embedding = torch.load(item_dict["extra_src_face_embedding_path"])
                item_dict["src_face_embedding"] = src_face_embedding
                del item_dict["extra_src_face_embedding_path"]
                return item_dict

            extra_cols = ["src_face_embedding_path"]
            dataset_kwargs = {"extra_cols": extra_cols, "callback": lambda x: x, "get_item_callback": get_item_callback}
            input_batch_dict = dict(
                prompt="input",
                src_face_embedding="src_face_embedding"
            )

            segments.append(f"face_guidance_scale{args.face_guidance_scale}")

        save_base = sampler.get_sample_save_dir(
            task=args.task, testset=Path(args.csv).stem, image_size=args.sample_image_size,
            segments=segments, rerank=args.rerank,
        )
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        logger.info(f"Save the generated images to: {save_base}")
        
        dataloader = sampler.build_sample_dataloader(
            data_source=dataset,
            save_template=save_template,
            batch_size=args.sample_batch_size,
            dataset_kwargs=dataset_kwargs,
            seed_type=args.seed_type,
            seed=args.seed,
            skip_exist=args.skip_exist,
        )

        sampler.batch_sample_general(
            dataloader=dataloader,
            run_fn=partial(sampler.batch_t2i, task=args.task),
            input_batch_dict=input_batch_dict,
            run_fn_kwargs=dict(
                size=args.sample_image_size,
                sequence_template=args.sequence_template,
            ),
            rerank=args.rerank,
        )
        # Print again at final for easy reading
        logger.info(f"Save the generated images to: {save_base}")

    elif args.task == "mmu":
        infer_kwargs = sampler.get_infer_kwargs()
        top_p = infer_kwargs["top_p"]
        top_k = infer_kwargs["top_k"]
        temperature = infer_kwargs["temperature"]
        segments = [f"t{temperature}_tp{top_p}_tk{top_k}"]
        save_base = sampler.get_sample_save_dir(
            task=args.task, testset=Path(args.csv).stem, segments=segments, subdir="mmu",
        )
        logger.info(f"Save the generated answers to: {save_base}")

        arrow_dataset_kwargs = dict(
            arrow_file=args.csv, seed_type=args.seed_type, seed=args.seed, save_template="{}",
            skip_exist=args.skip_exist, logger=logger,
        )

        column_dict = dict(image="image@bytes", prompt="prompt@default_const@Describe the image briefly.")
        image_processor = sampler.model_dict["image_processor"].preprocess
        dataset = ArrowDataset(column_dict=column_dict, image_processor=image_processor, **arrow_dataset_kwargs)

        dataloader = sampler.build_sample_dataloader(
            data_source=dataset,
            batch_size=args.sample_batch_size,
            skip_exist=args.skip_exist,
        )

        input_batch_dict = dict(prompt="prompt", image="image")
        sampler.batch_sample_general(
            dataloader=dataloader,
            run_fn=sampler.batch_mmu,
            input_batch_dict=input_batch_dict,
            save_dir=save_base,
        )
        # Print again at final for easy reading
        logger.info(f"Save the generated images to: {save_base}")

    else:
        raise NotImplementedError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
