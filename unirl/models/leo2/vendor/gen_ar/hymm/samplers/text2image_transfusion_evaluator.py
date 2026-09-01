import torch

from hymm.config import parse_eval_initial_args
from hymm.samplers.base_sampler import setup_distributed_initialize, setup_ptm_initialize
from hymm.samplers.text2image_transfusion_sampler import Text2ImageTransfusionSampler
from hymm.utils.file_utils import rank0_logger, safe_json

# Define a dummy placeholder for loading the PTM ckpt.
build_pretraining_data_loader = None

def main():
    initial_args, mode = parse_eval_initial_args()

    if mode != "ptm":
        world_size, rank, device = setup_distributed_initialize(initial_args, "deepspeed")
        logger = rank0_logger(rank)

        evaluator = Text2ImageTransfusionSampler.from_pretrained(
            ckpt_path=initial_args.ckpt,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )
        # Get updated args (include the yaml configs saved along with model checkpoint)
        args = evaluator.args
    else:
        # use extra_args to init megatron
        args = setup_ptm_initialize()
        logger = rank0_logger(torch.distributed.get_rank())
        evaluator = Text2ImageTransfusionSampler.ptm_from_pretrained(args, logger)

    evaluator.initialize_scores(args.evaluation_metrics)

    save_path = evaluator.get_metric_save_path(metric_type="score", image_size=args.metric_image_size)
    segments = [f"{args.denoise_type}{args.diff_infer_steps}", f"cfg{args.guidance_scale}"]
    image_save_base = (
        evaluator.get_sample_save_dir(testset=None, image_size=args.metric_image_size, segments=segments)
        if args.evaluation_metrics_save_image
        else None
    )
    kwargs = evaluator.get_infer_kwargs()

    evaluator.eval(
        image_size=args.metric_image_size,
        batch_size=args.metric_batch_size,
        save_path=save_path,
        rerank=args.rerank,
        image_save_base=image_save_base,
        extra_save_info=safe_json(kwargs),
    )


if __name__ == "__main__":
    main()
