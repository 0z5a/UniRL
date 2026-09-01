from hymm.config import parse_eval_initial_args
from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.samplers.text2image_ar_sampler import Text2ImageARSampler
from hymm.utils.file_utils import rank0_logger


def main():
    initial_args, _ = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, "deepspeed")
    logger = rank0_logger(rank)

    evaluator = Text2ImageARSampler.from_pretrained(
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
    # add logits processor config into segments automatically, cfg is included
    segments = []
    logits_processor_list = args.get("logits_processors_cfg", [])
    for logits_processor in logits_processor_list:
        for name, kwargs in logits_processor.items():
            if name in ["CfgLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"]:
                for k, v in kwargs.items():
                    seg = f"{k}{v}"
                    segments.append(seg)
    image_save_base = (
        evaluator.get_sample_save_dir(testset=None, image_size=args.metric_image_size, segments=segments)
        if args.evaluation_metrics_save_image
        else None
    )

    evaluator.eval(
        image_size=args.metric_image_size,
        batch_size=args.metric_batch_size,
        save_path=save_path,
        rerank=args.rerank,
        image_save_base=image_save_base,
    )


if __name__ == "__main__":
    main()
