from functools import partial
from typing import List

from hymm.config import parse_eval_initial_args
from hymm.constants import VISION_ENCODER_META_INFO
from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.samplers.multimodal_transfusion_sampler import MultimodalTransfusionSampler
from hymm.utils.file_utils import rank0_logger, safe_json


def recaption_processor(evaluator, inputs, seeds, generate_kwargs):
    prompt_list = [prompt + "<recaption>" for prompt in inputs]
    answers = evaluator.generate(
        prompts=prompt_list,
        seed=seeds,
        **generate_kwargs,
    )["samples"]
    return answers


def main():
    initial_args, _ = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, "deepspeed")
    logger = rank0_logger(rank)

    evaluator = MultimodalTransfusionSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = evaluator.args

    evaluator.initialize_scores(args.evaluation_metrics)
    kwargs = evaluator.get_infer_kwargs()

    save_path = evaluator.get_metric_save_path(metric_type="score", image_size=args.metric_image_size)
    segments = [
        f"{args.denoise_type}{args.diff_infer_steps}",
        f"cfg{args.guidance_scale}",
        f"t{kwargs['temperature']}_tp{kwargs['top_p']}_tk{kwargs['top_k']}",
    ]
    image_save_base = (
        evaluator.get_sample_save_dir(testset=None, image_size=args.metric_image_size, segments=segments)
        if args.evaluation_metrics_save_image
        else None
    )

    eval_kwargs = {}
    run_fn_kwargs = {}
    # Setup input processor before calling eval
    if args.task == "t2i":
        eval_kwargs["run_fn"] = evaluator.predict
        eval_kwargs["image_save_base"] = image_save_base

    elif args.task == "recap_t2i":
        eval_kwargs["input_processor"] = partial(recaption_processor, generate_kwargs=dict(
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
        ))
        eval_kwargs["image_save_base"] = image_save_base

    elif args.task == "image_captioning":
        vision_encoder_meta_info = VISION_ENCODER_META_INFO[args.vision_model_type]
        target_size = vision_encoder_meta_info["image_size"]
        eval_kwargs["dataset_kwargs"] = {
            "target_size": target_size,
            "pad_color": (127, 127, 127)
        }
        eval_kwargs["run_fn"] = evaluator.generate
        eval_kwargs["data_save_base"] = evaluator.get_sample_save_dir(segments=[segments[2]], subdir="mmu")
        # Pass to generate
        # kwargs["run_fn_kwargs"] = {"prompts": ""}
    
    elif args.task == "t2t":
        eval_kwargs["run_fn"] = evaluator.generate
        eval_kwargs["data_save_base"] = evaluator.get_sample_save_dir(segments=[segments[2]], subdir="lm")
        if "instruct" in initial_args.ckpt:
            def _formattor(cls, inputs_dict):
                SYSTEM_PROMPT = "User: {}\n\nAssistant: <answer>"
                prompts = inputs_dict.get('prompts')
                if prompts is None:
                    return inputs_dict
                if isinstance(prompts, List) and isinstance(prompts[0], str):
                    prompts = [SYSTEM_PROMPT.format(prompt) for prompt in prompts]
                elif isinstance(prompts, str):
                    prompts = SYSTEM_PROMPT.format(prompts)
                else:
                    return inputs_dict
                inputs_dict['prompts'] = prompts
                return inputs_dict

            eval_kwargs["input_processor"] = _formattor
        # Pass to generate
        run_fn_kwargs = {"return_first_pred_token_logits": True}

    else:
        raise ValueError(f"Task {args.task} not supported.")

    evaluator.eval(
        image_size=args.metric_image_size,
        batch_size=args.metric_batch_size,
        save_path=save_path,
        rerank=args.rerank,
        extra_save_info=safe_json(kwargs),
        run_fn_kwargs=run_fn_kwargs,
        **eval_kwargs,
    )


if __name__ == "__main__":
    main()
