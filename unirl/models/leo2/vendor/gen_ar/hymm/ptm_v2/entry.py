# Entry point for training PTM v2 models using different trainers.
# This script patches Megatron-LM for PTM v2 compatibility and dynamically
# loads and invokes the specified trainer.
# This entry script make sure that:
# 1. The necessary patches are applied before any Megatron-LM modules are imported.
# 2.

import sys
import argparse
import importlib

from .helpers import patch_megatron_for_ptm_v2


def get_trainer(name):
    assert '.' in name, (
        f"Invalid trainer name: {name}. A valid trainer name should be in the form of "
        f"<module_name>.<trainer_func>."
    )
    module_name, trainer_func = name.rsplit('.', 1)
    module_spec = importlib.import_module(f"hymm.ptm_v2.{module_name}")
    return getattr(module_spec, trainer_func)


def train():
    # Make sure to apply patchers before importing Megatron-LM modules
    patch_megatron_for_ptm_v2()

    # Parse ptm config yaml
    original_argv = sys.argv.copy()[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ptm-config-yaml", type=str, required=True, help="PTM config yaml file path")
    parser.add_argument("--trainer", type=str, required=True, help="Name of the trainer to use for training.")
    known_args, remaining_argv = parser.parse_known_args(original_argv)
    frozen_args = {}
    if known_args.ptm_config_yaml is not None:
        from hymm.core.arguments import parse_argv_from_yaml

        # config_argv will be handled by argparse later, frozen_args will be passed to args directly
        config_argv, frozen_args = parse_argv_from_yaml(
            known_args.ptm_config_yaml, allow_frozen=True, argv_overrides=remaining_argv
        )
        original_argv = config_argv + remaining_argv

    # Reformulate checkpointing args for handling manual/automatic resume
    from angelptm.megatron.training.arguments import maybe_replace_checkpointing_args_for_resume

    original_argv = maybe_replace_checkpointing_args_for_resume(original_argv)
    sys.argv = [sys.argv[0]] + original_argv

    # Invoke the specified trainer
    trainer = get_trainer(known_args.trainer)
    trainer(frozen_args)


if __name__ == "__main__":
    train()
