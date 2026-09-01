import warnings

import torch
from functools import partial

from hymm.config import parse_eval_initial_args
from hymm.constants import VISION_ENCODER_META_INFO
from hymm.samplers.base_sampler import setup_distributed_initialize, setup_ptm_initialize
from hymm.samplers.gemini_beta_sampler import GeminiBetaSampler
from hymm.utils.file_utils import rank0_logger, safe_json
from hymm.utils.torch_utils import except_collate_fn


warnings.filterwarnings("ignore", category=FutureWarning, module="transformer_engine")

# Define a dummy placeholder for loading the PTM ckpt.
build_pretraining_data_loader = None


def main():
    initial_args, mode = parse_eval_initial_args()
    if mode != 'ptm':
        world_size, rank, device = setup_distributed_initialize(initial_args, "deepspeed")
        logger = rank0_logger(rank)

        evaluator = GeminiBetaSampler.from_pretrained(
            ckpt_path=initial_args.ckpt,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )
        # Get updated args (include the yaml configs saved along with model checkpoint)
        args = evaluator.args
    # ptm inference
    else:
        # use extra_args to init megatron
        args = setup_ptm_initialize()
        logger = rank0_logger(torch.distributed.get_rank())
        evaluator = GeminiBetaSampler.ptm_from_pretrained(args, logger)

    evaluator.initialize_scores(args.evaluation_metrics)
    kwargs = evaluator.get_infer_kwargs()

    save_path = evaluator.get_metric_save_path(metric_type="score", image_size=args.metric_image_size)
    segments = [
        f"{args.denoise_type}{args.diff_infer_steps}",
        f"cfg{args.guidance_scale}",
        f"t{kwargs['temperature']}_tp{kwargs['top_p']}_tk{kwargs['top_k']}",
    ]

    eval_kwargs = {}
    custom_collate_fn = None
    # Setup input processor before calling eval
    if args.task == "t2i":
        save_image = True if args.evaluation_metrics_save_image else False
        run_fn = evaluator.batch_x2image
        run_fn_kwargs = dict(task='t2i', sample_image_size=args.metric_image_size)
        data_save_base = evaluator.get_sample_save_dir(
            image_size=args.metric_image_size, segments=segments[:2], subdir="samples") if save_image else None
        eval_kwargs["input_batch_dict"] = dict(
            batch_prompt_list="prompt",
        )

    elif args.task == "mmu":
        vision_encoder_meta_info = VISION_ENCODER_META_INFO[args.vision_model_type]
        if "image_size" in vision_encoder_meta_info:
            target_size = vision_encoder_meta_info["image_size"]
            eval_kwargs["dataset_kwargs"] = dict(target_size=target_size, pad_color=(127, 127, 127))
        else:
            eval_kwargs["dataset_kwargs"] = dict(target_size=None)

        if args.get("use_joint_image_feature"):

            def get_item_callback(item_dict):
                pil_image = item_dict["image"]
                image_info = evaluator.process_src_image(pil_image, args.metric_image_size, use_joint_image_feature=True)
                item_dict["image_info_list"] = [image_info]
                del item_dict["image"]
                return item_dict

            eval_kwargs["dataset_kwargs"]["get_item_callback"] = get_item_callback
            eval_kwargs["input_batch_dict"] = dict(
                prompt="prompt", batch_joint_image_info_list="image_info_list", is_dummy="is_dummy")
            custom_collate_fn = partial(except_collate_fn, except_keys=["references", "image_info_list"])
        else:
            eval_kwargs["input_batch_dict"] = dict(prompt="prompt", image="image")

        # Set verbose 2 to print the answers
        run_fn = evaluator.batch_x2text
        run_fn_kwargs = dict(xtype="mmu", verbose=args.verbose)
        data_save_base = evaluator.get_sample_save_dir(segments=[segments[2]], subdir="mmu")

    elif args.task == "lm":
        run_fn = evaluator.batch_x2text
        run_fn_kwargs = dict(xtype="lm", verbose=args.verbose)
        data_save_base = evaluator.get_sample_save_dir(segments=[segments[2]], subdir="lm")
        eval_kwargs["input_batch_dict"] = dict(prompt="prompt", is_dummy="is_dummy")
        eval_kwargs["dataset_kwargs"] = dict(tokenizer=args.tokenizer_name)

    else:
        raise ValueError(f"Task {args.task} not supported.")

    evaluator.eval(
        image_size=args.metric_image_size,
        batch_size=args.metric_batch_size,
        save_path=save_path,
        rerank=args.rerank,
        extra_save_info=safe_json(kwargs),
        run_fn=run_fn,
        run_fn_kwargs=run_fn_kwargs,
        data_save_base=data_save_base,
        collate_fn=custom_collate_fn,
        **eval_kwargs,
    )


if __name__ == "__main__":
    main()
