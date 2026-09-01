from hymm.config import parse_eval_initial_args
from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.samplers.visualtext2audio_ar_sampler import VisualText2AudioARSampler
from hymm.utils.file_utils import rank0_logger


def main():
    initial_args, _ = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, "deepspeed")
    logger = rank0_logger(rank)
    evaluator = VisualText2AudioARSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = evaluator.args

    evaluator.initialize_scores(args.evaluation_metrics)
    save_path = evaluator.get_metric_save_path(args.metric_image_size, "score")
    save_base = args.sample_save_path
    # add logits processor config into segments automatically, cfg is included
    evaluator.eval(
        batch_size=args.metric_batch_size,
        save_path=save_path,
        save_base=save_base,
    )


if __name__ == "__main__":
    main()
