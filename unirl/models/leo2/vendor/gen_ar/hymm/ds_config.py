import argparse


def get_deepspeed_config(args: argparse.Namespace):
    deepspeed_config = {
        "train_batch_size": args.global_batch_size,
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": args.log_every,
        "optimizer": {
            "type": args.optimizer_name,
            "params": dict(**args.optimizer_params),
        },
        "gradient_clipping": args.gradient_clipping,
        "prescale_gradients": True,
        "fp16": {
            "enabled": args.precision == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.zero_stage,
            "reduce_scatter": False,
            "reduce_bucket_size": args.reduce_bucket_size,
            "overlap_comm": args.overlap_comm,
        },
    }
    return deepspeed_config
