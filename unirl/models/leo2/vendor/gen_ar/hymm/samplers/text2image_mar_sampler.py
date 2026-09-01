from hymm.ar import DDPMPipeline
from hymm.config import parse_eval_initial_args
from hymm.diffusion import load_scheduler
from hymm.models.text_encoder import TextEncoder
from hymm.samplers.text2image_mlm_sampler import Text2ImageMLMSampler
from hymm.samplers.base_sampler import (
    setup_distributed_initialize,
    text2image_interactive,
    text2image_batch,
)
from hymm.utils.file_utils import rank0_logger, safe_json


class Text2ImageMARSampler(Text2ImageMLMSampler):
    @classmethod
    def build_extra_model(cls, args, model_dict, factor_kwargs, logger=None):
        model_dict = super().build_extra_model(args, model_dict, factor_kwargs, logger)

        diff_scheduler = load_scheduler(args)
        diff_pipeline = DDPMPipeline(
            model=model_dict['model'].diffloss.net,
            scheduler=diff_scheduler,
        )
        diff_pipeline.set_progress_bar_config(disable=True)
        model_dict['pipeline'] = diff_pipeline

        return model_dict


class Text2ImageMARTESampler(Text2ImageMLMSampler):
    @classmethod
    def build_extra_model(cls, args, model_dict, factor_kwargs, logger=None):
        model_dict = super().build_extra_model(args, model_dict, factor_kwargs, logger)

        diff_scheduler = load_scheduler(args)
        diff_pipeline = DDPMPipeline(
            model=model_dict['model'].diffloss.net,
            scheduler=diff_scheduler,
        )
        diff_pipeline.set_progress_bar_config(disable=True)
        model_dict['pipeline'] = diff_pipeline

        text_encoder = TextEncoder(
            text_encoder_type="t5",
            max_length=256,
            text_encoder_precision="bf16",
            tokenizer_type="t5",
            use_attention_mask=False,
            infer_mode="encoder",
            logger=logger,
            device=factor_kwargs['device'],
        )
        model_dict['text_encoder'] = text_encoder

        return model_dict


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)

    print(f"Launch sampler with mode: {mode}")
    print(f"Distributed environment: world_size: {world_size}, rank: {rank}, device: {device}")

    logger = rank0_logger(rank)

    sampler = Text2ImageMARSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args

    # Start sampling or evaluation
    if args.evaluation_metrics:
        sampler.initialize_scores(args.evaluation_metrics)

        save_path = sampler.get_metric_save_path(args.metric_image_size, "score")
        image_save_base = (
            sampler.get_sample_save_dir(testset=None, image_size=args.metric_image_size)
            if args.evaluation_metrics_save_image
            else None
        )
        kwargs = sampler.get_infer_kwargs()

        sampler.eval(
            image_size=args.metric_image_size,
            batch_size=args.metric_batch_size,
            save_path=save_path,
            image_save_base=image_save_base,
            extra_save_info=safe_json(kwargs),
            **kwargs,
        )

    elif args.interactive:
        text2image_interactive(args, sampler, logger)

    elif args.csv is not None:
        text2image_batch(args, args.csv, sampler, logger)

    else:
        raise ValueError("Please provide `--evaluation-metrics`, `--interactive` (with `--prompt`), or `--csv`.")


if __name__ == "__main__":
    main()
