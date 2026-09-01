# -*- coding: utf-8 -*-
# from pipeline_emu2_gen at BAAI EMU2

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# from PIL import Image
import numpy as np
import torch
# from torchvision import transforms as TF
from tqdm import tqdm
import time
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor

from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderKL
# from transformers import CLIPImageProcessor
# from transformers import AutoModelForCausalLM # , AutoTokenizer   ## we do not use text in the first version

# from hymm.models.diffusion.vqunet import VQUNet
# from hymm.models.autoencoders.emu2.modeling_emu import EmuForCausalLM
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.utils.file_utils import log_in_safe_logger

EVA_IMAGE_SIZE = 448
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
# DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"

@dataclass
class EmuVisualGenerationPipelineOutput(BaseOutput):
    samples: Union[List[Any], np.ndarray]
    latents_tensor: torch.Tensor
    images_np: np.ndarray


class ClipToken2ImagePipeline(DiffusionPipeline):

    def __init__(
        self,
        diffusion_model: ModelMixin,
        scheduler: EulerDiscreteScheduler,
        vae: AutoencoderKL,
        # tokenizer: AutoTokenizer,
        multimodal_encoder: "EmuForCausalLM",
        args = None,
        logger=None,
        input_range: str = "01", # "01" , "-11"or "normal"
        output_range: str = "01", # "01" , "-11"
    ):
        super().__init__()
        self.register_modules(
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            vae=vae,
            # tokenizer=tokenizer,
            multimodal_encoder=multimodal_encoder,
            logger=logger
        )
        self.args = args
        self.logger = logger
        self._saved_logger = logger
        self.input_range = input_range
        self.output_range = output_range
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        

        self.negative_prompt = {}

    def device(self, module):
        return next(module.parameters()).device

    def dtype(self, module):
        return next(module.parameters()).dtype

    def set_logging_enabled(self, enabled=True):
        """Temporarily enable/disable logging"""
        self.logger = self._saved_logger if enabled else None
        self.diffusion_model.set_logging_enabled(enabled=enabled)
        # self.multimodal_encoder: EmuForCausalLM
        self.multimodal_encoder.set_logging_enabled(enabled=enabled)

    def vit_encode(self, inputs, do_classifier_free_guidance, device, dtype):
        # 1. Encode input prompt
        times_point = {}
        times_length = {}
        times_point['start_time'] = time.time()
        
        prompt_embeds = self._prepare_and_encode_inputs(
            inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
        ).to(dtype).to(device)
        
        times_point['encode_image_prompt'] = time.time()
        times_length['encode_image_prompt'] = times_point['encode_image_prompt'] - times_point['start_time']
        
        quantized_prompt, emb_loss, ind, other_info = self.diffusion_model.quantizer_encode(prompt_embeds)
        
        times_point['quantizer_encode'] = time.time()
        times_length['quantizer_encode'] = times_point['quantizer_encode'] - times_point['encode_image_prompt']
        log_in_safe_logger(times_length, self.logger, "ClipToken2ImagePipeline.vit_encode: time to encode image_prompt and quantizer_encode ")
        
        return quantized_prompt, emb_loss, ind, other_info

    def diffusion_decode(self,
                          quantized_prompt,
                          dtype, device, infer_steps, guidance_scale, output_type,
                          crop_info=None, original_size=None, sample_size=None, 
                          height=None, width=None,
                          ):
        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = quantized_prompt.shape[0] // 2 if do_classifier_free_guidance else quantized_prompt.shape[0]
        if do_classifier_free_guidance: assert quantized_prompt.shape[0] == batch_size * 2, f"quantized_prompt shape: {quantized_prompt.shape} must match batch_size: {batch_size}"
        dequantized_prompt = self.diffusion_model.quantizer_decode(quantized_prompt)


        unet_added_conditions = {}
        
        # if crop_info is None, use the default time_ids in the model
        
        if crop_info is not None:
            time_ids, height, width = self.diffusion_model.prepare_time_ids(
                crop_info, 
                original_size, 
                sample_size, 
                quantized_prompt.shape[0],
                # do_classifier_free_guidance,
                device
            )
        else:
            assert height is not None and width is not None, "height and width must be provided if crop_info is None"
            time_ids = self.diffusion_model.prepare_default_time_ids(quantized_prompt.shape[0], height, width)
        
        assert time_ids.shape[0] == dequantized_prompt.shape[0], f"time_ids shape: {time_ids.shape} must match dequantized_prompt shape: {inputs.shape}"

        unet_added_conditions["time_ids"] = time_ids
        unet_added_conditions["text_embeds"] = torch.mean(dequantized_prompt, dim=1)

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(infer_steps, device=device)
        timesteps = self.scheduler.timesteps
        log_in_safe_logger(timesteps, self.logger, "timesteps in pipeline of ClipToken2ImagePipeline")

        # 3. Prepare latent variables

        shape = (
            batch_size,
            self.diffusion_model.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(shape, device=device, dtype=dtype)
        # latents in edm start from N(0, 13.1585)
        latents = latents * self.scheduler.init_noise_sigma

        # 4. Denoising loop
        ## precision control copy from token2imagepipeline; line 576
        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (target_dtype != torch.float32)
        
        timesteps_tobe_logged = timesteps[::(len(timesteps)//2+1)]
        for t in timesteps:
            # 2B x 4 x H x W
            should_log = (t in timesteps_tobe_logged)
            if should_log:  
                log_in_safe_logger(
                    t,
                    self.logger,
                    f"Log at timestep {t} in pipeline of {timesteps_tobe_logged}",
                )
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                # todo move quantizer out of the loop
                
                # self.diffusion_model: VQUNet
                self.diffusion_model.set_logging_enabled(enabled= should_log)
                noise_pred = self.diffusion_model(
                    latent_model_input,
                    t,
                    cond=None, # we prequantize the prompt
                    added_cond_kwargs=unet_added_conditions, # necessary for sdxl
                    cond_hat = dequantized_prompt,
                )["x"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # clamped_noise_pred = torch.clamp(noise_pred, -3, 3)
            # # or just skip to avoid artifacts
            clamped_noise_pred = noise_pred
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(clamped_noise_pred, t, latents).prev_sample
            if should_log:  
                # noise_pred is a normal distribution N(0, 1). Range [-3, 3] should cover 99.7% of the elements
                log_in_safe_logger(noise_pred, self.logger, f"Time step {t}: the previous guided noisy sample with cfg {guidance_scale} is:")
                num_elements_out_of_range = ((noise_pred < -3).logical_or(noise_pred > 3)).sum().item() / noise_pred.numel() *100.0
                if self.logger is not None: self.logger.info(f"Number of elements out of [-3, 3] after cfg: { num_elements_out_of_range :.3f}%")
                log_in_safe_logger(noise_pred,  self.logger,    f"noise_pred after cfg at timestep {t}")
                log_in_safe_logger(latents,     self.logger,    f"the updated noisy sample latents at timestep {t}")

        log_in_safe_logger(latents, self.logger, f"latents after the denoising loop in pipeline of ClipToken2ImagePipeline")

        # noise_pred is a normal distribution N(0, 1). Range [-3, 3] should cover 99.7% of the elements
        # calculate how many elements out of [-3, 3]
        clamp_range = 1.0
        num_elements_out_of_range = ((latents < -clamp_range).logical_or(latents > clamp_range)).sum().item() / latents.numel() *100.0
        if self.logger is not None: self.logger.info(f"Number of latents elements out of [-{clamp_range}, {clamp_range}] after pipeline: {num_elements_out_of_range:.3f}%")
        ## EXP RESULT: naive clamping is harmful for image quality # latents = torch.clamp(latents, -clamp_range, clamp_range)

        # 5. Post-processing
        # Expose the decode process to support output_type
        latents = 1 / self.vae.config.scaling_factor * latents
        torch.cuda.empty_cache()
        with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
            n, c, h, w = latents.shape
            # Calculate max tokens per batch based on image size
            # VAE decoding will produce images 8x larger in each dimension
            tokens_per_image = h * w
            
            # Maximum tokens for A800-80 GPU is 128k (batch size 8 * height 128 * width 128)
            MAX_TOKENS = 4 * 128 * 1024  # 128k tokens
            
            # Calculate safe chunk size based on image resolution

            chunk_size = max(1, min(1, MAX_TOKENS // tokens_per_image))
            
            if self.logger is not None:
                self.logger.info(f"Using chunk size {chunk_size} for images of size {h}x{w} to control tokens under {MAX_TOKENS}")
            
            image = None
            for i in range(0, n, chunk_size):
                current_chunk_size = min(chunk_size, n - i)
                chunk = latents[i:i + current_chunk_size]
                image_chunk = self.vae.decode(chunk).sample
                
                if image is None:
                    image = image_chunk
                else:
                    image = torch.cat([image, image_chunk], dim=0)
            log_in_safe_logger(
                image,
                self.logger,
                f"image after decode in pipeline of ClipToken2ImagePipeline",
            )
        # return 
        do_denormalize = [bool(self.output_range == "01")] * image.shape[0]
        images_output_type = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)
        log_in_safe_logger(
            images_output_type,
            self.logger,
            f"images_output_type after postprocess in pipeline of ClipToken2ImagePipeline",
        )
        images_np = self.image_processor.postprocess(
            image, output_type='np', do_denormalize=do_denormalize)    

        # 6. Run safety checker
        # images, has_nsfw_concept = self.run_safety_checker(images)

        # 7. Convert to PIL
        return EmuVisualGenerationPipelineOutput(
            samples = images_output_type,
            latents_tensor = latents,
            images_np = images_np
        )

    @torch.no_grad()
    def __call__(
        self,
        condition: torch.Tensor = None,
        sample_size = [1024, 1024],
        infer_steps: int = 50,
        guidance_scale: float = 3.,
        crop_info: List[int]  = [0, 0], # list or torch.Size([4, 2]) 
        original_size: List[int] = [1024, 1024], # list or torch.Size([4, 2])
        # other args
        output_type: str = "pil",
        should_log: bool = True,
        **model_input_extra_kwargs,
    ):
        """Emu2 pipeline for image generation.

        Args:
            condition (torch.Tensor, optional): The input image tensor. shape: [1, 3, H, W], min: 0.00, max: 1.00, mean: 0.45, std: 0.27, first element: 0.79. Defaults to None.
            sample_size (list, optional): The size of the image (not latent) sample. Defaults to [1024, 1024].
            infer_steps (int, optional): The number of inference steps. Defaults to 50.
            guidance_scale (float, optional): The guidance scale. Defaults to 3..
            crop_info (List[int], optional): The crop information. Defaults to [0, 0].
            original_size (List[int], optional): The original size of the image. Defaults to [1024, 1024].
            output_type (`str`, *optional*, defaults to `pil`):
                The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.

        Returns:
            _type_: _description_
        """
        # translate hunyuan namespace to emu2 namespace
        inputs = condition
        batch_size = inputs.shape[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if self.logger is not None: self.logger.info(f"Classifier-free guidance: {guidance_scale}")
        log_in_safe_logger(inputs, self.logger, "inputs in pipeline of ClipToken2ImagePipeline")

        device = self.device(self.diffusion_model)
        dtype = self.dtype(self.diffusion_model)
                

        quantized_prompt, emb_loss, ind, other_info = self.vit_encode(
            inputs, do_classifier_free_guidance, device, dtype, should_log=should_log
        )

        return_dict = self.diffusion_decode(
            quantized_prompt,
            dtype, device, infer_steps, guidance_scale, output_type,
            crop_info=crop_info, original_size=original_size, sample_size=sample_size, 
            should_log=should_log
        )
        return return_dict

    @torch.no_grad()
    def check_input(self, image, min_max_range = [0, 1]):
        n, c, h, w =image.shape
        logger = self.logger
        min_range, max_range = min_max_range
        if image.min() < min_range or image.max() > max_range:
            if logger is not None: logger.warning(f"Input image range should be in [{min_range}, {max_range}], but got {image.min(), image.max()}")
            else: print(f"Input image range should be in [{min_range}, {max_range}], but got {image.min(), image.max()}")

    def _prepare_and_encode_inputs( # done
        self,
        inputs,
        do_classifier_free_guidance: bool = False,
        # placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        times_length = {}
        times_point = {}
        times_point['start_time'] = time.time()

        device = self.device(self.multimodal_encoder.model.visual)
        dtype = self.dtype(self.multimodal_encoder.model.visual)
        
        inputs = inputs.type(dtype).to(device)
        
        if self.input_range == "01":
            self.check_input(inputs, min_max_range=[0, 1])
        elif self.input_range == "-11":
            self.check_input(inputs, min_max_range=[-1, 1])
        else:
            raise ValueError(f"Input range {self.input_range} is not supported")
        
        # has_image, has_text = False, False
        # text_prompt, image_prompt = "", []
        if isinstance(inputs, list):
            image_prompt = []
            if self.logger is not None: self.logger.info(f"Assume input has range in [0, 1], transform to normal distribution with mean 0 and std 1")
            for x in inputs:
                image_prompt.append(self.multimodal_encoder.model.transform(x, self.input_range))

            if self.logger is not None: self.logger.info(f"Input shape before stack image: {image_prompt[0].shape=}")
            image_prompt = torch.stack(image_prompt)
        else:
            image_prompt = self.multimodal_encoder.model.transform(inputs, self.input_range)
        image_prompt = image_prompt.type(dtype).to(device)

        log_in_safe_logger(image_prompt, self.logger, "image_prompt before encode_image in ClipToken2ImagePipeline._prepare_and_encode_inputs ")
        times_point['transform_image_range'] = time.time()
        times_length['transform_image_range'] = times_point['transform_image_range'] - times_point['start_time']
        prompt = self.multimodal_encoder.model.encode_image(image=image_prompt)
        times_point['encode_image_prompt'] = time.time()
        times_length['encode_image_prompt'] = times_point['encode_image_prompt'] - times_point['transform_image_range']
        
        if do_classifier_free_guidance:
            key = "[NULL_IMAGE]"
            negative_image = torch.zeros_like(image_prompt)
            self.negative_prompt[key] = self.multimodal_encoder.model.encode_image(image=negative_image)
            log_in_safe_logger(self.negative_prompt[key], self.logger, "Negative prompt batch for classifier-free guidance")
            prompt = torch.cat([prompt, self.negative_prompt[key]], dim=0)
            times_point['encode_negative_prompt'] = time.time()
            times_length['encode_negative_prompt'] = times_point['encode_negative_prompt'] - times_point['encode_image_prompt']
        times_point['encode_both_prompt'] = time.time()
        times_length['encode_both_prompt'] = times_point['encode_both_prompt'] - times_point['start_time']
        log_in_safe_logger(times_length, self.logger, "ClipToken2ImagePipeline._prepare_and_encode_inputs: time to encode image_prompt and negative_prompt ")
        ## if has text, use LLM to encode text and image
        ## to avoid confusion, we do not use text in the first version
        # else:
        #     prompt = self.multimodal_encoder.generate_image(text=[text_prompt], image=image_prompt, tokenizer=self.tokenizer)
        #     if do_classifier_free_guidance:
        #         key = ""
        #         if key not in self.negative_prompt:
        #             self.negative_prompt[key] = self.multimodal_encoder.generate_image(text=[""], tokenizer=self.tokenizer)
        #         prompt = torch.cat([prompt, self.negative_prompt[key]], dim=0)

        return prompt




