from hymm.config import parse_eval_initial_args
from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.samplers.text2image_mlm_sampler import Text2ImageMLMSampler
from hymm.utils.file_utils import rank0_logger, safe_json


def main():
    initial_args, _ = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, "deepspeed")
    logger = rank0_logger(rank)

    evaluator = Text2ImageMLMSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = evaluator.args

    evaluator.initialize_scores(args.evaluation_metrics)

    save_path = evaluator.get_metric_save_path(metric_type="score", image_size=args.metric_image_size)
    image_save_base = (
        evaluator.get_sample_save_dir(testset=None, image_size=args.metric_image_size)
        if args.evaluation_metrics_save_image
        else None
    )
    logger.info(f"metric save_path: {save_path}")
    logger.info(f"image_save_base: {image_save_base}")
    kwargs = evaluator.get_infer_kwargs()

    evaluator.eval(
        image_size=args.metric_image_size,
        batch_size=args.metric_batch_size,
        save_path=save_path,
        image_save_base=image_save_base,
        extra_save_info=safe_json(kwargs),
        **kwargs,
    )


if __name__ == "__main__":
    main()
