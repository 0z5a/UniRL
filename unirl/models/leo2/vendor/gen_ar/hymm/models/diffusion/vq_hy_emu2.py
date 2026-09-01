#%%
""" from https://github.com/CompVis/stable-diffusion/, remove all the training related parts, e.g. discriminator
"""

from copy import deepcopy
import os
import sys
from pathlib import Path
import datetime 

dir_path = "/apdcephfs_cq8/share_2938211/chenyangqi/hunyuan_multimoda_gen_ar_image_dev"
sys.path.append(dir_path)
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from contextlib import contextmanager

from hymm.models.autoencoders.modules.vector_quantize_pytorch_lucidrains.vector_quantize_pytorch import VectorQuantize as VectorQuantizer_lucidrains
from hymm.utils.file_utils import log_in_safe_logger, safe_check_grad
from hymm.utils.file_utils import safe_dir



class VQModel(nn.Module):
    def __init__(
        self,
        vq_type = 'lucidrains', # 'hy' or 'lucidrains'
        ddconfig = {'z_channels': 1792}, # dimension of evaclip
        n_embed=16384,
        embed_dim=256, # not sure what can be the best parameter; 4, 8, 64 ...
        ckpt_path=None,
        codebook_ckpt_path=None,
        ignore_keys=[],
        ema_update=False,
        use_quant_conv=True,
        use_quant_norm=False,
        use_cosine_sim = False,
        logger=None,
        skip_codebook= False, # just for debug
        args=None,
        cache_state_dict=False,
        learnable_codebook=False,
        **kwargs,
    ):
        super().__init__()
        self.vq_type = vq_type
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.codebook_size = n_embed
        self.skip_codebook = skip_codebook
        self.logger = logger
        self._saved_logger = logger
        self.args = args
        self.cache_state_dict = cache_state_dict
        self.ema_update = ema_update


        if self.vq_type == 'lucidrains':
            self.use_cosine_sim = use_cosine_sim
            # refactor following code to first build a dictionary of argument, then pass it to the VectorQuantizer_lucidrains class
            self.quantize = VectorQuantizer_lucidrains(
                dim=embed_dim, 
                codebook_size=n_embed, 
                codebook_dim=embed_dim, 
                accept_image_fmap=True,
                kmeans_init=True, 
                kmeans_iters=10,     
                use_cosine_sim = use_cosine_sim,
                codebook_ckpt_path=codebook_ckpt_path, # FIXME correct
                logger=logger,
                ema_update=ema_update,
                learnable_codebook=learnable_codebook
            )
        else:
            print(f"VQModel: {vq_type} not implemented")
            raise NotImplementedError()

        self.use_quant_conv = use_quant_conv
        self.use_quant_norm = use_quant_norm
        if use_quant_conv:
            self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
            if logger is not None: logger.info(f"Quant conv from {ddconfig['z_channels']} to {embed_dim}")
            if use_quant_norm:
                if logger is not None: logger.info(f"Using quant norm with {embed_dim}")
                self.quant_norm = nn.LayerNorm(embed_dim)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        
        # why we need so many ignore keys? how about filter it in the begining, or only involve it in th last init_from_ckpt?
        if ckpt_path is not None: # None
            self.logger.info(f"Loading whole model ckpt from {ckpt_path=}, code book will use ckpt {codebook_ckpt_path} if provided")
            if codebook_ckpt_path is not None:
                ignore_keys=["quantize._codebook"] 
                self.logger.info(f"Ignore keys: {ignore_keys}")
            else:
                ignore_keys=[]
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            self.logger.warning(f"There is no checkpoint path provided in VQModel __init__")
        # print modules in log
        if self.logger is not None:
            self.logger.info(f"Initialized vq type: {self.vq_type}")
            self.logger.info(f"{str(self)}")

    @classmethod
    def from_config(cls, json_cfg_path):
        import json
        print("Loading from config")
        if isinstance(json_cfg_path, str) and os.path.exists(json_cfg_path):
            print("Loading from file")
            with open(json_cfg_path, "r") as f:
                cfgs = json.load(f)
            return cls(**cfgs["params"])
        else:
            print("Loading from dict")
            return cls(**json_cfg_path["params"])
        
    # This may overwrite the codebook in lucidrains's VectorQuantizer
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    if self.logger is not None:
                        self.logger.info(f"Deleting key {k} from state_dict.")
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
        if self.logger is not None:
            self.logger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            self.logger.info(f"Missing Keys: {missing}")
            self.logger.info(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        log_in_safe_logger(x, self.logger, "VQModel encode: Input x")

        h = x
        if self.logger is not None: 
            self.logger.info(f"use quant_conv: {self.use_quant_conv}; use quant_norm: {self.use_quant_norm}")
        if self.use_quant_conv:
            h = self.quant_conv(h)
        if self.use_quant_norm:
            shape_n, shape_c, shape_h, shape_w = h.shape
            h_nlp = rearrange(h, "n c h w -> n (h w) c")
            h_nlp_normed   = self.quant_norm(h_nlp) # NLP style normalization,  torch.norm(h_nlp_normed, dim=-1) = ones(n, (hw))
            h = rearrange(h_nlp_normed, "n (h w) c -> n c h w", h=shape_h, w=shape_w, c=shape_c)

        log_in_safe_logger(h, self.logger, "encode: Hidden space after potential quant_conv quant_norm")  
            

        if self.skip_codebook: # only for debug
            quant, emb_loss, embed_ind, other_info =  h, torch.tensor(0.0), None, None
        else:
            if self.vq_type == 'lucidrains':
                # do not use torch autocast here, because it will cause the codebook to be float16
                # conver h to type of self.quantize.codebook
                input_dtype = h.dtype
                h = h.to(self.quantize.codebook.dtype)
                    
                quantize, embed_ind, loss = self.quantize(h, logger=self.logger)
                n_quantize, d_quantize, h_quantize, w_quantize = quantize.shape
                if self.use_cosine_sim:
                    dequantize_scale = d_quantize**0.5
                    quantize = quantize * dequantize_scale # FIXME rescale for cosine sim is better to merge in vector quantizer not outside

                quant = quantize.to(input_dtype)
                embed_ind = embed_ind
                emb_loss = loss
                other_info = None
            else:
                print(f"VQModel: {self.vq_type} not implemented")
                raise NotImplementedError() 
                # safe_check_grad("quantize", quantize, self.logger)
                # safe_check_grad(" self.quantize(h).loss", loss, self.logger)
        return quant, emb_loss, embed_ind, other_info

    def encode_to_prequant(self, x):
        # h = self.encoder(x)
        h = x
        h = self.quant_conv(h)
        return h
    # @torch.no_grad()
    def vq_encode(self, x):
        # _, _, [_, _, indices] = self.encode(x)
        # return indices
        quant, emb_loss, embed_ind, other_info = self.encode(x)
        return embed_ind

    def decode(self, quant):
        if self.use_quant_conv:
            quant = self.post_quant_conv(quant)
        # dec = self.decoder(quant)
        dec = quant
        return dec

    # @torch.no_grad()
    def vq_decode(self, code_b):
        if len(code_b.shape) > 2:
            b, h, w = code_b.shape
            code_b = code_b.reshape(1, -1)
        else:
            b, c = code_b.shape
            h = w = int(np.sqrt(c))
            assert h * w == c, "c not compatible with h * w, c={} h={} w={}".format(c, h, w)
        quant_b = self.quantize.get_codebook_entry(code_b, None)
        quant_b = quant_b.view(b, h, w, self.embed_dim)
        quant_b = rearrange(quant_b, "b h w c -> b c h w")
        dec = self.decode(quant_b)
        return dec

    def set_logging_enabled(self, enabled=True):
        """Temporarily enable/disable logging"""
        # self._saved_logger = self.logger
        self.logger = self._saved_logger if enabled is True else None

    def forward(self, input, return_dict=False):
        for k, v in self.quantize.named_parameters():
            log_in_safe_logger(v, self.logger, f"{k}")
        quant, diff, ind, other_info = self.encode(input)
        dec = self.decode(quant)

        state_dict ={
            "input": input,
            "quantize.embedding.weight": self.quantize.codebook,
            "quant": quant,
            "diff": diff,
            "ind": ind,
            "dec": dec,
            "other_info": other_info
        }

        for k, v in state_dict.items():
            if self.logger is not None:
                # Log the state dict item
                log_in_safe_logger(v, self.logger, f"forward: {k}")

                # Special handling for quant tensor to log its norm
                if k == "quant":
                    rearrange_quant = rearrange(v, 'b c h w -> b h w c')
                    norm_quant = torch.norm(rearrange_quant, dim=-1)
                    log_in_safe_logger(norm_quant, self.logger, "forward: norm of quant")
            
        if self.cache_state_dict:
            # get node, rank, world size
            if torch.distributed.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()  # Rank of the current process in the cluster.
                device = rank % torch.cuda.device_count()  # Device of the current process in current node.
            else:
                world_size = 1
                rank = 0
                device = 0
            # save this dict to a file
            # FIXME: Add args to itialization to support cache_state_dict
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.args.get("exp_dir", None) is not None:
                exp_dir = self.args.exp_dir
            elif self.args.get("save_folder", None) is not None:
                exp_dir = self.args.save_folder
            else:
                exp_dir = "./trash"
            exp_dir = exp_dir.replace("log_EXP", "log_EXP_sh7")
            save_folder = safe_dir(safe_dir(exp_dir) / "quantization_state_dict")
            if self.logger is not None:
                self.logger.info(f"Save state_dict to {save_folder}")
            save_path = f"{save_folder}/vq_state_shape{quant.shape}_rank{rank}in{world_size}_{time_str}.pth"
            torch.save(state_dict, save_path)
        
        if return_dict :
            return state_dict
        else:
            return dec, diff, ind



class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        # h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        # dec = self.decoder(quant)
        dec = quant
        return dec





class IdentityVQModel(torch.nn.Module):
    def __init__(self, *args, vq_interface=True, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        quant, emb_loss, embed_ind, other_info = x, torch.tensor(0.0), None, None
        return quant, emb_loss, embed_ind, other_info

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x, None, None
    def set_logging_enabled(self, enabled=True):
        return None



if __name__ == "__main__":
    from omegaconf import OmegaConf
    yml = "/apdcephfs_cq8/share_2938211/chenyangqi/hunyuan_multimoda_gen_ar_image_dev/hymm/configs/emu2/ldm-vq-f8_hy.yaml"
    cfg = OmegaConf.load(yml)

    model = VQModel.from_config(cfg["model"])
    print(model)
    test_img = "/apdcephfs_cq8/share_2938211/chenyangqi/hunyuan_multimoda_gen_ar_image_dev/datasets/face_words/test_image_256x256.png"
    from PIL import Image
    import numpy as np
    from torchvision.transforms import Compose, ToTensor, Normalize

    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    img = Image.open(test_img).convert("RGB").resize((512, 512))
    img = transform(img).unsqueeze(0)
    # random feature tensor with shape (1, 256, 64, 64)
    img = torch.randn(1, 1700, 8, 8)
    img_2 = torch.randn(1, 1700, 8, 8)
    
    
    with torch.no_grad():
        # quant, emb_loss, [(perplexity, min_encodings, min_encoding_indices)]
        z, emb_loss, [perplexity, min_encodings, min_encoding_indices] = model.encode(img)
        xrec = model.decode(z)
    print(xrec.shape)
    
    # mse error between original and reconstructed image
    mse = F.mse_loss(xrec, img)
    mse_random = F.mse_loss(img_2, img)

