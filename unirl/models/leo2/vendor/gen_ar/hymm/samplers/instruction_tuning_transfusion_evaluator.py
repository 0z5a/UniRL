import os
import ipdb
from hymm.config import parse_eval_initial_args
from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.samplers.instruction_tuning_transfusion_sampler import InstructionTuningTransfusionSampler
from hymm.utils.file_utils import rank0_logger


def main():
    initial_args, _ = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, "deepspeed")
    logger = rank0_logger(rank)

    evaluator = InstructionTuningTransfusionSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = evaluator.args

    segments = [f"{args.denoise_type}{args.diff_infer_steps}", f"textcfg{args.guidance_scale}"]
    
    if args.get("face_guidance_scale", None) is not None:
        segments.append(f"facecfg{args.face_guidance_scale}")

    evaluator.args.metric_save_file_suffix = "_".join(segments)
    # json file save path
    save_path = evaluator.get_metric_save_path(metric_type="score", image_size=args.metric_image_size)
    
    # First "" is for testset name
    image_save_base = (
        evaluator.get_sample_save_dir(testset="id", segments=["{}"]+segments)
        if args.evaluation_metrics_save_image
        else None
    )
    logger.info(f"image_save_base: {image_save_base}")
    if args.instruction_tuning_task == "id":
        input_batch_dict = {"instruction_list": "prompt", "src_img_tensor_batch": "src_image"}
        if args.get("src_condition_type", None) != None and "face_embed" in args.get("src_condition_type", None):
            input_batch_dict["src_face_embedding"] = "src_face_embedding"
    else:
        raise NotImplementedError(f"Instruction tuning task {args.instruction_tuning_task} not implemented")

    # Loading evaluation metrics models to GPU
    evaluator.initialize_scores(args.evaluation_metrics)
    evaluator.eval(
        image_size=args.metric_image_size,
        batch_size=args.metric_batch_size,
        save_path=save_path,
        rerank=args.rerank,
        image_save_base=image_save_base,
        input_batch_dict=input_batch_dict,
        task=args.instruction_tuning_task
    )


if __name__ == "__main__":
    print(f"DEBUG: {os.environ.get('DEBUG', 'false')}")
    print(f"To enable DEBUG, Set DEBUG to true in cmd using export DEBUG=true")
    if os.environ.get("DEBUG", "false") == "true":
        print("Entering DEBUG mode and traceback will be printed")
        try:
            main()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred: {e}")
            ipdb.post_mortem()  # Enter the debugger
    else:
        main()
