import json
from typing import List, Tuple, Optional, Union, Dict
from einops import rearrange

import os
import torch
import torch.nn as nn

from diffusers import UNet2DConditionModel
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from safetensors.torch import load_file

from hymm.models.diffusion.vq_hy_emu2 import VQModel, IdentityVQModel
from hymm.models.diffusion.attn_layers import BasicAttentionLayer
from hymm.utils.file_utils import log_in_safe_logger



VQUNet_CONFIG = {  
    "VQ-SDXL-EMU2": {
        "pretrained_diffusion_model_path": "/apdcephfs_sh7/share_301124792/chenyangqi/pretrained_ckpt/BAAI/Emu2-Gen/unet",}, # Attn+MLP / Total    ~2B /  ~2.5B
    "VQ-SDXL-EMU2-CPU": {
        "pretrained_diffusion_model_path": "/apdcephfs_cq8/share_2938211/chenyangqi/pretrained_ckpt/BAAI/Emu2-Gen/unet",}, # Attn+MLP / Total    ~2B /  ~2.5B
}


class VQUNet(ModelMixin, ConfigMixin):
    """
    VQUNet model first generate a dicrete token from CLIP token, then generate the image using diffusion model.
    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.
    """

    @register_to_config
    def __init__(
        self,
        args,  # not used here; will be used to pass some other model-irrelevant configs, such as gradient_checkpoint
        model_config,
        logger = None,
        dtype: Optional[torch.dtype] = None,  # FIXME, this arg is not used in the model
        device: Optional[
            torch.device
        ] = None,
        # required by pipleine
        in_channels=4,
        sample_latent_size=128, # 1024 / 8 = 128 for SDXL
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.logger = logger
        self._saved_logger = logger
        self.model_config = model_config
        
        if 'VQModel' in model_config:
            # self.quantizer will quantize input clip token and return a tenosr with same shape
            if self.logger is not None: 
                self.logger.info(
                f"Initialize VQModel using Config {model_config.VQModel}"
            )
            self.quantizer = VQModel(logger=logger, **model_config.VQModel, args=args)

            
        else:
            # Set as Identity layer for load 172_8_8 emu ckpt or debugging
            self.quantizer = nn.Identity(**factory_kwargs)
            # self.quantizer = IdentityVQModel(**factory_kwargs)

        # The render from quantized token to image latent
        # set as UNet in the beginning; will be updated to a more complex model, e.g, DIT

        if self.logger is not None: 
            self.logger.info(
                f"Start Initialize UNet2DConditionModel from pretrained {model_config.pretrained_diffusion_model_path} using config.json"
            )
        model_config_json = model_config.pretrained_diffusion_model_path + '/config.json'
        with open(model_config_json, 'r') as f:
            model_config_dict = json.load(f)
        # FIXME the name is conflicted with the diffusion model in pipeline
        self.diffusion_model = UNet2DConditionModel(**model_config_dict)
        # if unet.pt exists, load the state dict
        if os.path.exists(model_config.pretrained_diffusion_model_path + '/unet.pt'):
            if self.logger is not None: 
                self.logger.info(
                f"Start Load UNet2DConditionModel state dict from {model_config.pretrained_diffusion_model_path} + '/unet.pt'"
            )
            self.diffusion_model.load_state_dict(torch.load(model_config.pretrained_diffusion_model_path + '/unet.pt', map_location=lambda storage, loc: storage))
            if self.logger is not None: 
                self.logger.info(
                    f"Load UNet2DConditionModel state dict from {model_config.pretrained_diffusion_model_path} + '/unet.pt' done"
                )
        else:
            if self.logger is not None: 
                self.logger.info(
                    f"No unet.pt found in {model_config.pretrained_diffusion_model_path}, skip loading"
                )
        
        if self.logger is not None: 
            self.logger.info(f"Move model to {device}, convert to {dtype}")
        if dtype is not None:
            self.quantizer.to(dtype)
            self.diffusion_model.to(dtype)
        if device is not None and device != -1:
            self.quantizer.to(device)
            self.diffusion_model.to(device)
        if self.logger is not None: 
            self.logger.info("VQUNet is initialized")

    # @staticmethod
    def load_state_dict(self, state_dict, ignore_keys=list(), strict=None):
        
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
        return missing_keys, unexpected_keys
    def quantizer_encode(self, cond: torch.Tensor):
        cond_2d = self.unpatchify_nlc_to_nchw(cond)
        
        quant, diff, ind, other_info = self.quantizer.encode(cond_2d)
        return quant, diff, ind, other_info
        
    def quantizer_decode(self, quant):
        dec = self.quantizer.decode(quant)
        
        dequantized_cond = rearrange(dec, "n c h w -> n (h w) c")

        return dequantized_cond
    
    def unpatchify_nlc_to_nchw(self, cond: torch.Tensor):
        # cond shape is [2, 64, 1792])        
        # reshape from (n, hw, c) to (n, c, h, w)
        n, hw, c = cond.shape
        h = w = int(hw ** 0.5)
        assert h * w == hw, f"{hw=}, {h=}, {w=}"
        cond_2d = rearrange(cond, "n (h w) c -> n c h w", h=h, w=w, c=c)
        return cond_2d


    def quantizer_encode_decode(self, cond: torch.Tensor):
        """CLIP Token to low dimensional token."""
        log_in_safe_logger(
            cond, 
            self.logger,
            "Before quantizer forward condition",

        )
        
        cond_2d = self.unpatchify_nlc_to_nchw(cond)

        if 'VQModel' in self.model_config:
            quant_return = self.quantizer(cond_2d, return_dict=True)
            cond_hat_2d, emb_loss, ind, other_info = quant_return['dec'], quant_return['diff'], quant_return['ind'], quant_return["other_info"]
        else:
            quant_return = self.quantizer(cond_2d)
            cond_hat_2d, emb_loss, ind, other_info = quant_return, 0.0, None, None
            
        cond_hat = rearrange(cond_hat_2d, "n c h w -> n (h w) c")
        
        if self.logger is not None:
            self.logger.info(
                f"After quantizer forward: \t"
                f"cond_hat shape: {cond_hat.shape}, "
                f"Codebook loss: {emb_loss}, "
                f"ind: {ind}",
                f"other_info: {other_info}"
            )
        return cond_hat, emb_loss, ind, other_info



    def set_logging_enabled(self, enabled=True):
        """Temporarily enable/disable logging"""
        self.logger = self._saved_logger if enabled else None
        if hasattr(self.quantizer, 'set_logging_enabled'):
            self.quantizer.set_logging_enabled(enabled= enabled)
    
    def forward(
        self,
        x: torch.Tensor,  # noisy latents
        t: torch.Tensor,  # Should be in range(0, 1000).
        cond: torch.Tensor = None,
        added_cond_kwargs: dict = {},
        return_dict: bool = True,
        # should_log=True,
        cond_hat: torch.Tensor = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if cond_hat is None:
            cond_hat, emb_loss, ind, other_info = self.quantizer_encode_decode(cond)
        else:
            cond_hat = cond_hat
            emb_loss, ind, other_info = torch.tensor(0.0), None, None
            
        log_in_safe_logger(
            added_cond_kwargs, 
            self.logger,
            "Before model forward added_cond_kwargs",
        )
        diffusion_out = self.diffusion_model(
            x,
            t,
            encoder_hidden_states=cond_hat,
            added_cond_kwargs=added_cond_kwargs,
        )

        log_in_safe_logger(
            diffusion_out.sample,
            self.logger,
            f"After VQUNet forward, the predicted latents are:",
        )
        if return_dict:
            out_dict = {}
            out_dict["x"] = diffusion_out.sample
            out_dict["emb_loss"] = emb_loss
            return out_dict
        else:
            return diffusion_out.sample

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

    def prepare_default_time_ids(self, n, h, w):
        original_size = [h, w]
        sample_size  = [h, w]     
        height, width = sample_size
        crop_info = [0, 0]
        time_ids = torch.LongTensor(original_size + crop_info + [height, width]).to(self.device)
        # time_ids = torch.repeat_interleave(time_ids[None, ...], n, dim=0) # bug here?
        time_ids = torch.stack([time_ids] * n, dim=0)
        return time_ids

    def prepare_time_ids(
        self,
        crop_info,
        original_size,
        sample_size,
        batch_size,
        # do_classifier_free_guidance,
        device
    ):
        """Prepare time IDs for the diffusion process.
        
        Args:
            crop_info: List[int] or tensor of shape (batch_size, 2)
            original_size: List[int] or tensor of shape (batch_size, 2)
            sample_size: List[int] or tensor of shape (batch_size, 2)
            batch_size: int
            do_classifier_free_guidance: bool
            device: torch.device
        
        Returns:
            dict: Contains time_ids and other conditions for the UNet
        """
        # Convert lists to tensors if necessary
        if isinstance(crop_info, (list, tuple)):
            crop_info = torch.stack([torch.LongTensor(crop_info)] * batch_size, dim=0)
        if isinstance(original_size, (list, tuple)):
            original_size = torch.stack([torch.LongTensor(original_size)] * batch_size, dim=0)
        if isinstance(sample_size, int):
            sample_size = [sample_size, sample_size]
        if isinstance(sample_size, (list, tuple)):
            sample_size = torch.stack([torch.LongTensor(sample_size)] * batch_size, dim=0)
        
        # Validate sample size dimensions
        assert sample_size.shape[1] == 2, "sample_size should have shape (b, 2)"
        for i in range(sample_size.shape[0]):
            assert sample_size[i, 0] == sample_size[0, 0], "h should be the same"
            assert sample_size[i, 1] == sample_size[0, 1], "w should be the same"
        
        # Prepare added conditions for UNet
        time_ids = torch.cat([original_size, crop_info, sample_size], dim=1).to(device)
        
        # if do_classifier_free_guidance:
        #     time_ids = torch.cat([time_ids, time_ids], dim=0)
        # else:
        #     time_ids = time_ids
            
        return time_ids, sample_size[0, 0], sample_size[0, 1]
