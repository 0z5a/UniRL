import numpy as np
import os
from PIL import Image
from pathlib import Path
import ipdb
from tqdm import tqdm
from glob import glob
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
from hymm.samplers.cliptoken2image_diffusion_sampler import ClipToken2ImageDiffusionSampler

def setup_distributed_cpu_inference(args):
    # check whether gpu is available
    gpu_available = torch.cuda.is_available()
    if gpu_available:   
        world_size, rank, device = setup_distributed_inference(args, mode=args.get('mode', "none"))
        size = 1024
    else:
        world_size, rank, device = 1, 0, 'cpu'
        args.model_name = 'VQ-SDXL-EMU2-CPU'
        args.ckpt = None
        args.load_key = '.'
        size = 512
    print(f"world_size: {world_size}, rank: {rank}, device: {device}, model_name: {args.model_name}")
    return world_size, rank, device

def main(args, logger, world_size, rank, device):
    # test  sampler without loading ckpt

    # ======================== Initialize experiment directory and logger =========================
    if dist.is_initialized():
        dist.barrier() 

    # if args.ddp:
    if dist.is_initialized():
        dist.barrier() 

    logger.info(f"Rank {rank}/{world_size};")
    logger.info(f"Batch size per GPU: {args.get('batch_size', 2)}")
    
    # ======================== Initialize sampler =========================
    evaluator = ClipToken2ImageDiffusionSampler.from_pretrained(
        args=args,
        ckpt_path=args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = evaluator.args

    evaluator.initialize_scores(args.evaluation_metrics)
    # image size is 1024 FIXME: move to config
    if args.metric_save_path is None:
        args.metric_save_path = evaluator.get_metric_save_path(1024, "score")
    if args.sample_save_path is None:
        args.sample_save_path = evaluator.get_sample_save_dir(testset="clip2image_folder", image_size=1024) / "samples"
    # else:
    # save_path = evaluator.get_metric_save_path(1024, "score")
    # add logits processor config into segments automatically, cfg is included
    # segments = []
    # image size is 1024 FIXME: move to config
    image_save_base = (
        args.sample_save_path
        if args.evaluation_metrics_save_image
        else None
    )
    evaluator: ClipToken2ImageDiffusionSampler
    evaluator.eval(
        image_size=1024,
        batch_size=args.metric_batch_size,
        save_path=args.metric_save_path,
        image_save_base=image_save_base,
    )

    if dist.is_initialized():
        dist.barrier()  # Synchronize processes before exiting


if __name__ == "__main__":

    try:
        args = parse_args()
        
        
        world_size, rank, device = setup_distributed_cpu_inference(args)
        if args.ckpt.endswith(".pt"):
            exp_dir_str, logger = ClipToken2ImageDiffusionSampler.init_exp_dir_logger(args, rank)

            args.metric_save_path = str(Path(exp_dir_str) / "evaluation" / "score.json")
            args.sample_save_path = str(Path(exp_dir_str) / "samples")
            logger.info(f"Evaluating {args.ckpt}")
            main(args, logger, world_size, rank, device)
        else:
            exp_folder = args.ckpt
            args.exp_dir = exp_folder
            exp_dir_str, logger = ClipToken2ImageDiffusionSampler.init_exp_dir_logger(args, rank)
            
            checkpoints_folder = os.path.join(exp_folder, "checkpoints")
            samples_folder = os.path.join(exp_folder, "samples")
            evaluation_folder = os.path.join(exp_folder, "evaluation")
            # if not end with .pt, then it is a folder
            
            ckpt_list = sorted(glob(checkpoints_folder + "/*/mp_rank_00_model_states.pt"))[::-1]

            logger.info(f"Found {len(ckpt_list)} checkpoints in {checkpoints_folder}")
            num_ckpt_to_eval = 3
            # sample num_ckpt_to_eval in ckpt_list
            interval = max(len(ckpt_list) // num_ckpt_to_eval, 1)
            ckpt_to_eval_list = ckpt_list[0::interval]
            logger.info(f"Evaluating {len(ckpt_to_eval_list)} checkpoints")
            
            for ckpt in tqdm(ckpt_to_eval_list):
                logger.info(f"Evaluating {ckpt}")
                args.ckpt = ckpt # args use ckpt but sampler use ckpt_path
                args.metric_save_path = None
                args.sample_save_path = None
                main(args, logger, world_size, rank, device)


    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
        ipdb.post_mortem()  # Enter the debugger
