import numpy as np
import os
from PIL import Image
from pathlib import Path

import torch
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader

from hymm.config import parse_args
from hymm.models import load_vae, build_model
from hymm.models.diffusion.posemb_layers import get_nd_rotary_pos_embed
from hymm.utils.file_utils import rank0_logger
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.diffusion import load_diffusion_pipeline
from hymm.diffusion.pipelines.cliptoken2image_pipeline import ClipToken2ImagePipeline
from hymm.samplers.base_sampler import setup_distributed_initialize as setup_distributed_inference
from hymm.samplers.base_sampler import BaseSampler, cliptoken2image_batch
from hymm.utils.file_utils import safe_read_file_cache

class ClipToken2ImageDiffusionSampler(BaseSampler):
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

        self.model_dtype = PRECISION_TO_TYPE[args.precision]
        self.autocast_enabled = self.model_dtype in [torch.half, torch.bfloat16]
        self.logger = logger
        
        self.pipeline = load_diffusion_pipeline(
            args=self.args, # args.denoise_type == "euler"
            rank=self.rank,
            pipeline_name="cliptoken2image",
            diffusion_model=self.model_dict["model"],
            vae=self.model_dict["vae"],
            multimodal_encoder=self.model_dict["multimodal_encoder"],
            # feature_extractor=self.model_dict["feature_extractor"],
            device=self.device,
            logger = logger,
        )

    @staticmethod
    def build_extra_model(args, factor_kwargs, logger=None):
        device = factor_kwargs["device"]
        vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=device,
            logger=logger,
        )
        multimodal_encoder = load_vae(
            args.clipvision_type,
            args.clipvision_precision,
            device=device,
            logger=logger,
        )
        
        return {
            "vae": vae,
            "multimodal_encoder": multimodal_encoder,
        }
    
    @torch.no_grad()
    def predict(self, 
                image, 
                size=(1024, 1024), 
                crop_info=[0, 0], 
                original_size=[1024, 1024], 
                output_type="pil", # 'pt', 'pil' or 'np'
                **kwargs,
                ):
        """
        pipeline_out = self.pipeline(image, sample_size, infer_steps, guidance_scale)
        Args:
            image (torch.Tensor): shape: [1, 3, H, W], min: 0.00, max: 1.00, mean: 0.45, std: 0.27, first element: 0.79
            sample_size (tuple, optional): The size of the image sample. Defaults to (1024, 1024).
            output_type (`str`, *optional*, defaults to `pil`):
                The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.

        Returns:
            dict: A dictionary containing the generated images. "samples"
            todo: add more metrics, codebook utilization, test loss
        """
        # Ideally, we need to conduct vector quantization here, not in pipeline
        # FIXME: has a lot of redundancy in the args; 
        # merge the sampler logic with trainer into model file
        guidace_scale = self.args.get("guidance_scale", 3.0)
        infer_steps = self.args.get("infer_steps", 50)        
        # add IDE type hint of pipeline type to clip2image_diffusion_pipeline
        self.pipeline: ClipToken2ImagePipeline
        pipeline_out = self.pipeline(
            condition=image,
            sample_size=size, # Do not use size as API name; ambiguity of token / latent / image sample size
            infer_steps=infer_steps,
            guidance_scale=guidace_scale,   
            crop_info = crop_info,
            original_size = original_size,
            output_type=output_type,
            # model_input_extra_kwargs={
            #     "freqs_cos": freqs_cos,
            #     "freqs_sin": freqs_sin,
            # }
        )

        return pipeline_out
    
        

    @classmethod
    def from_pretrained(cls, args, ckpt_path, rank=0, world_size=1, device=0, logger=None):
        """
        Initialize the sampling pipeline.

        Args:
            args (dict or EasyDict): The arguments for the inference pipeline.
            ckpt_path (str or pathlib.Path): The checkpoint path.
            rank (int): The rank for distributed inference. # The rank of the current process / GPU in the cluster. Default is 0.
            world_size (int): The world size for distributed inference. # The total number of processes / GPUs across all nodes. Default is 1.
            device (int): The device for inference. # The device of the current process in current node. Default is 0.
            logger (logging.Logger): The logger for the inference pipeline. Default is None.
        """
        if logger is None:
            from loguru import logger

        # ======================== Get the checkpoint path =======================
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
            if ckpt_path.is_dir():
                if (ckpt_path / "checkpoints").exists():
                    latest_name = (ckpt_path / "checkpoints" / "latest").read_text().strip()
                    ckpt_path = ckpt_path / "checkpoints" / latest_name
                try:
                    ckpt_path = next(ckpt_path.glob("*_model_states.pt"))
                except StopIteration:
                    raise FileNotFoundError(f"No model state found in the directory: {ckpt_path}")
            if ckpt_path.is_file():
                assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
            else:
                raise FileNotFoundError(f"Invalid checkpoint path: {ckpt_path}")
            logger.info(f"Get checkpoint path: {ckpt_path}")

            # ============================= Load the config ===========================
            args = cls.load_config(args, ckpt_path, logger)

            # =========================== Build main model ===========================
            logger.info("Building model...")
            factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
            model, model_settings = build_model(args, logger=logger)
            model = model.to(**factor_kwargs)

            # cache_ckpt = safe_read_file_cache(args.ckpt, "log_EXP", "log_EXP_sh7", logger, rank) # this is not currect, args.ckpt != ckpt_path
            cache_ckpt = ckpt_path            
            logger.info(f"Loading model from ckpt_path: {cache_ckpt} ({args.load_key}); ignore_keys: {args.get('ckpt_ignore_keys', None)}")
            cls.load_state_dict(model, cache_ckpt, args.load_key, ignore_keys=args.get("ckpt_ignore_keys", None))
            logger.info(f"Loaded.")
        else:
            logger.info(f"{ckpt_path=}, assume it is a new model or will be initialized during build_model.") 
            
            # =========================== Build main model ===========================
            logger.info("Building model without ckpt initialization...")
            factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
            model, model_settings = build_model(args, logger=logger)
            model = model.to(**factor_kwargs)
        
        model.requires_grad_(False)
        model.eval()

        # =========================== Build extra model ===========================
        model_dict = cls.build_extra_model(args, factor_kwargs, logger)

        model_dict["model"] = model
        model_dict["model_settings"] = model_settings

        return cls(
            args=args,
            model_dict=model_dict,
            ckpt_path=ckpt_path,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )

class FlatImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = Path(folder)
        self.transform = transform
        self.image_paths = sorted(list(self.folder.glob("*.png")))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)  # Return path as second element instead of class

def main():
    # test  sampler without loading ckpt

    args = parse_args()
    # check whether gpu is available
    gpu_available = torch.cuda.is_available()
    if gpu_available:   
        world_size, rank, device = setup_distributed_inference(args, mode="ddp" if args.ddp else "none")
        size = 1024
    else:
        world_size, rank, device = 1, 0, 'cpu'
        args.model_name = 'VQ-SDXL-EMU2-CPU'
        args.ckpt = None
        args.load_key = '.'
        size = 512
    print(f"world_size: {world_size}, rank: {rank}, device: {device}, model_name: {args.model_name}")


    # ======================== Initialize experiment directory and logger =========================
    if args.ddp:
        dist.barrier() 
    sample_save_path, logger = ClipToken2ImageDiffusionSampler.init_exp_dir_logger(args, rank)
    if args.ddp:
        dist.barrier() 

    logger.info(f"Rank {rank}/{world_size};")
    logger.info(f"Batch size per GPU: {args.get('batch_size', 2)}")
    logger.info(f"Loading model from pretrained ckpt: {args.ckpt=} ({args.load_key=})")
    
    # ======================== Initialize sampler =========================
    diffusion_sampler = ClipToken2ImageDiffusionSampler.from_pretrained(
        args=args,
        ckpt_path=args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )

    # Use the batch wrapper or explicit manual loop
    cliptoken2image_batch(args, diffusion_sampler, logger)

    if args.ddp:
        dist.barrier()  # Synchronize processes before exiting


if __name__ == "__main__":
    import ipdb
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
        ipdb.post_mortem()  # Enter the debugger
