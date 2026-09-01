import argparse
import os

ENTRIES = ["deepspeed", "ptm", "pure_torch", "ptm_v2"]


def _set_default_cuda_device_early():
    """Bind this process to its own GPU as early as possible.

    Frameworks call torch.cuda.set_device(local_rank) only after arg parsing /
    config loading / heavy imports. Any default-device (cuda:0) CUDA op that
    runs before that point creates a CUDA context on local GPU0, wasting 0.5GB*7
    on local GPU0.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank))
    except Exception:
        pass


if __name__ == "__main__":
    launcher_parser = argparse.ArgumentParser(description="Hunyuan Multimodal Training Launcher", add_help=False)
    launcher_parser.add_argument("--entry", default="deepspeed", choices=ENTRIES,
                                 help="Define the launcher entry point.")
    known_args, remaining_args = launcher_parser.parse_known_args()

    if 'ptm' not in known_args.entry:
        try:
            if 'CUDA_DEVICE_MAX_CONNECTIONS' not in os.environ or int(os.environ['CUDA_DEVICE_MAX_CONNECTIONS']) <= 1:
                os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '64'
        except ValueError:
            os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '64'
        # import hy_parallelism earlier
        import hy_parallelism

    _set_default_cuda_device_early()

    # =======================================================================================
    # Route to different training entry points. We support four entries:
    # 1. "deepspeed": the native Hymm training entry point.
    # 2. "ptm": the AngelPTM training entry point.
    # 3. "pure_torch": the pure PyTorch Hymm training entry point.
    # 4. "ptm_v2": the AngelPTM v2 training entry point with Megatron-LM patched for PTM v2.
    # =======================================================================================

    # Deprecated, Deepspeed
    if known_args.entry == "deepspeed":
        from hymm.config import parse_args
        from hymm.trainers import get_trainer

        args = parse_args()
        trainer = get_trainer(args)
        trainer.train()

    # Deprecated, AngelPTM v1
    elif known_args.entry == "ptm":
        from hymm.ptm.train_gemini_beta import train

        train()

    # Recommended, Pure Torch (FSDP support)
    elif known_args.entry == "pure_torch":
        from hymm.trainers.entry import train

        train()

    # Recommended, AngelPTM v2 (mcore support)
    elif known_args.entry == "ptm_v2":
        from hymm.ptm_v2.entry import train

        train()

    else:
        raise ValueError(f"Unknown entry: {known_args.entry}, choose from {ENTRIES}.")
