import json
from typing import List, Tuple, Optional, Union, Dict
from einops import rearrange

import os
import torch
import torch.nn as nn

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models import ModelMixin
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import AutoConfig

from safetensors.torch import load_file

from hymm.constants import VAE_META_INFO
# from hymm.models.diffusion.vq_hy_emu2 import VQModel
from hymm.models.autoencoders.emu2.modeling_emu import  EmuForCausalLM
from hymm.models.autoencoders.modules.distributions import DiagonalGaussianDistribution
from hymm.models.diffusion.attn_layers import BasicAttentionLayer
from hymm.models.diffusion.vqunet import VQUNet
from hymm.utils.file_utils import log_in_safe_logger, safe_file
# from hymm.diffusion import load_diffusion_pipeline
from hymm.diffusion.pipelines.cliptoken2image_pipeline import ClipToken2ImagePipeline


from easydict import EasyDict

VQUNet_CONFIG = {  
    "VQ-SDXL-EMU2": {
        "pretrained_diffusion_model_path": "/apdcephfs_sh7/share_301124792/chenyangqi/pretrained_ckpt/BAAI/Emu2-Gen/unet",}, # Attn+MLP / Total    ~2B /  ~2.5B
    "VQ-SDXL-EMU2-CPU": {
        "pretrained_diffusion_model_path": "/apdcephfs_cq8/share_2938211/chenyangqi/pretrained_ckpt/BAAI/Emu2-Gen/unet",}, # Attn+MLP / Total    ~2B /  ~2.5B
}


class EVACLIP_256x16x16_SDXL(ModelMixin, ConfigMixin):
    """
    VQUNet model first generate a dicrete token from CLIP token, then generate the image using diffusion model.
    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.
    """

    @register_to_config
    def __init__(
        self,
        args,  # not used here; will be used to pass some other model-irrelevant configs, such as gradient_checkpoint
        model_config,
        vae_path =None,
        logger = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        shift_factor = 0.0,
        scaling_factor = 1.0,
        _trans_type = "-11", # "-11" or "01"
        guidance_scale = 1.5,
        return_dict = False,
        return_tensor_shape = "fbchw", # "bchw" or "fbchw"
        # required by pipleine
        # in_channels=4,
        # sample_latent_size=128, # 1024 / 8 = 128 for SDXL
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.logger = logger
        self._saved_logger = logger
        self.model_config = EasyDict(model_config)
        self.pretrained_diffusion_model_path = str(vae_path) + "/sdxl_256x16x16_dec"
        self.model_config.pretrained_diffusion_model_path = self.pretrained_diffusion_model_path
        
        self.shift_factor = shift_factor
        self.scaling_factor = scaling_factor
        self.config.shift_factor = shift_factor
        self.config.scaling_factor = scaling_factor
        self._trans_type = _trans_type
        self.guidance_scale = guidance_scale
        self.return_dict = return_dict
        self.return_tensor_shape = return_tensor_shape
        # There are three configs;
        # 1. SDXL VAE config                            VAE_META_INFO
        # 2. Vector Quantization Diffusion model        self.model_config and vae_path
        # 3. VIT encoder                                vae_path

        factory_kwargs = {"device": device, "dtype": dtype}
        ########################################################## SDXL VAE #########################################################
        self.vae_sdxl = AutoencoderKL.from_pretrained(
            VAE_META_INFO["88-vae-sdxl"]["path"],
            variant = 'bf16',
            use_safetensors=True,
        )
        self.vae_sdxl: AutoencoderKL

        ######################################################### Vector Quantization #########################################################

        if 'VQModel' in self.model_config:
            assert 'codebook_ckpt_path' not in self.model_config.VQModel, "This is a continuous model. Codebook_ckpt_path should not be in model_config.VQModel"
            
        self.diffusion_model = VQUNet(
            args=args,
            model_config=self.model_config,
            logger=logger, 
            dtype=dtype,
            device=device,
        )
        self.quantized_prompt_uncond = torch.load(safe_file(vae_path) / "quantized_prompt_ZERO_NULL_IMAGE.pt", map_location=lambda storage, loc: storage).to(device).to(dtype)

        diffusion_model_state_dict = torch.load(safe_file(vae_path) / 'vqunet_119k.pt', map_location=lambda storage, loc: storage)
        missing, unexpected = self.diffusion_model.load_state_dict(diffusion_model_state_dict)
        if len(missing) > 0:
            self.logger.info(f" VQUNetDiffusion Model Missing Keys: {missing}")
        if len(unexpected) > 0:
            self.logger.info(f" VQUNet Diffusion Model Unexpected Keys: {unexpected}")
 
        ######################################################### VIT Encoder #########################################################
        # FIXME start
        evaclip_path = safe_file(vae_path) / "evaclip_emu2"
        logger.info(f"Load evaclip config from {evaclip_path}")
        config = AutoConfig.from_pretrained(
            f"{evaclip_path}",
            trust_remote_code=True
        )
        config.pooling_stride = self.model_config.get("pooling_stride", None)
        logger.info(f"Start Init evaclip model from config, set pooling_stride to {config.pooling_stride}")

        self.vit_encoder = EmuForCausalLM(config, logger=logger)
        ckpt_path = evaclip_path / "evaclip.pt"
        logger.info(f"Load evaclip checkpoint start, from {ckpt_path}")
        self.vit_encoder.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
        logger.info(f"Load evaclip checkpoint done, from {ckpt_path}")
        self.vit_encoder: EmuForCausalLM
        ########################################################## Pipeline #########################################################

        # create a diffusion pipeline FIXME check mar
        self.scheduler = EulerDiscreteScheduler(  
            beta_end =  0.012,
            beta_schedule =  "scaled_linear",
            beta_start =  0.00085,
            interpolation_type =  "linear",
            num_train_timesteps =  1000,
            prediction_type =  "epsilon",
            steps_offset =  1,
            timestep_spacing =  "leading",
            trained_betas =  None,
            use_karras_sigmas =  False,
        )


        # FIXME check mar
        self.pipeline = ClipToken2ImagePipeline(
            args=self.args,
            diffusion_model=self.diffusion_model,
            scheduler=self.scheduler,
            vae=self.vae_sdxl,
            multimodal_encoder=self.vit_encoder,
            logger = self.logger,
            input_range=self._trans_type,
            output_range=self._trans_type,
            # **extra_model_dict
        )
        self.pipeline = self.pipeline.to(device)
        self.set_logging_enabled(enabled=False)


    @classmethod
    def from_config(cls, config_file, args, logger=None, **kwargs):
        vae_folder = os.path.dirname(config_file)
        
        # load json
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        logger = args.logger if logger is None else logger
        if args is None:
            args = {"model_precision": "bf16"}
            args = EasyDict(args)
        return cls( args=args, model_config=config_dict, vae_path=vae_folder, logger=logger, **kwargs)
    
    # @staticmethod
    def load_state_dict(self, state_dict, ignore_keys=list(), strict=None):
        # FIXME start is this used?
        if self.logger is not None: 
            self.logger.info(f"[WARNING] strict={strict} is not used in load_state_dict")
            self.logger.info(f"ignore_keys: {ignore_keys}")
        keys = state_dict.keys()
        filtered_state_dict = state_dict.copy()
        keys = list(filtered_state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    if self.logger is not None:
                        self.logger.info(f"Deleting key {k} from state_dict.")
                    del filtered_state_dict[k]
        filtered_state_keys = filtered_state_dict.keys()
        if self.logger is not None:
            # compare keys and filtered_state_keys, log the keys in keys but not in filtered_state_keys
            for k in keys:
                if k not in filtered_state_keys:
                    self.logger.info(f"Key {k} in state_dict but not in filtered_state_dict.")

        missing_keys, unexpected_keys = super().load_state_dict(filtered_state_dict, strict=False)
        if self.logger is not None: self.logger.info(f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")



    def encode(self, image: torch.Tensor, **kwargs):
        """CLIP Token to latent token."""
        log_in_safe_logger(image, self.logger, f"EVACLIP_256x16x16_SDXL encode image: ")
        quantized_prompt, emb_loss, ind, other_info = self.pipeline.vit_encode(
            image,
            do_classifier_free_guidance=False,
            device=self.device,
            dtype=self.dtype,
            **kwargs
        )
        posterior = DiagonalGaussianDistribution(torch.cat([quantized_prompt, quantized_prompt], dim=1), deterministic=True)
        return AutoencoderKLOutput(latent_dist=posterior)

    def vq_encode(self, image: torch.Tensor, **kwargs):
        return self.encode(image, **kwargs)
    
    def decode(self, quantized_prompt, 
               return_dict=None,
               generator=None,
               **kwargs):
        # crop_info = self.diffusion_model.
        # better to set as config in pipeline
        infer_steps = 32
        guidance_scale = self.guidance_scale
        
        output_type = "pt"
        height = 1024
        width = 1024
        if return_dict is None:
            return_dict = self.return_dict
        N, C, H, W = quantized_prompt.shape

        quantized_prompt = torch.cat([quantized_prompt, self.quantized_prompt_uncond.repeat(N, 1, 1, 1)], dim=0)
        pipeline_return_dict_result = self.pipeline.diffusion_decode(
            quantized_prompt, 
            self.dtype, self.device, infer_steps, guidance_scale, output_type,
            height=height, width=width
        )
        log_in_safe_logger(pipeline_return_dict_result, self.logger, f"EVACLIP_256x16x16_SDXL decode image: ")
        # Logic here can be improved : pipeline_return_dict_result[0].shape = torch.Size([1, 3, 1024, 1024])
        if return_dict:
            return pipeline_return_dict_result
        else:
            samples = pipeline_return_dict_result["samples"]
            if self.return_tensor_shape == "bchw" and len(samples.shape) == 4:
                return samples # torch.Size([1, 3, 1024, 1024])
            elif self.return_tensor_shape == "fbchw" and len(samples.shape) == 4:
                return torch.unsqueeze(samples, 0) # torch.Size([1, 3, 1024, 1024])
            else:
                raise ValueError(f"Invalid return_tensor_shape: {self.return_tensor_shape}, expected 'bchw' or 'fbchw', got {self.return_tensor_shape}")
    
    def vq_decode(self, latents, **kwargs):
        return self.decode(latents, **kwargs)


    def set_logging_enabled(self, enabled=True):
        """Temporarily enable/disable logging"""
        self.logger = self._saved_logger if enabled else None
        self.pipeline.set_logging_enabled(enabled=enabled)
        self.vit_encoder.set_logging_enabled(enabled=enabled)

    def params_count(self):
        # TEST ME
        counts = {
            "total": sum(p.numel() for p in self.parameters()),
            "attn+mlp": sum(
                p.numel()
                for name, p in self.named_parameters()
                if "attn" in name or "mlp" in name
            ),
            "quantizer": sum(p.numel() for p in self.quantizer.parameters()),
        }
        return counts
    
    def enable_deterministic(self):
        # set deterministic to True for all attention layers
        
        has_attn_layer = False
        for name, module in self.diffusion_model.named_modules():
            if isinstance(module, BasicAttentionLayer):
                module.enable_deterministic()
                has_attn_layer = True
        if not has_attn_layer:
            self.logger.warning("No attention layer found in the model, deterministic is not enabled")
