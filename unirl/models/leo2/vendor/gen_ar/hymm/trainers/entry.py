import argparse
import importlib
import sys

from hymm.core.arguments import parse_argv_from_yaml
from hymm.config import add_core_args, validate_args


def get_trainer(name):
    assert '.' in name, (
        f"Invalid trainer name: {name}. A valid trainer name should be in the form of "
        f"<module_name>.<trainer_cls>."
    )
    module_name, trainer_cls = name.rsplit('.', 1)
    module_spec = importlib.import_module(f"hymm.trainers.{module_name}")
    return getattr(module_spec, trainer_cls)


def train():
    # Parse config yaml
    original_argv = sys.argv.copy()[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-path", type=str, required=True, help="Config yaml file path")
    parser.add_argument("--trainer", type=str, required=True, help="Name of the trainer to use for training.")
    known_args, remaining_argv = parser.parse_known_args(original_argv)

    # config_argv will be handled by argparse later, frozen_args will be passed to args directly
    config_argv, frozen_args = parse_argv_from_yaml(
        known_args.config_path, allow_frozen=True, argv_overrides=remaining_argv
    )
    sys.argv = [sys.argv[0]] + config_argv + remaining_argv

    # parse args
    parser = argparse.ArgumentParser(description="Hunyuan Multimodal Pure Torch Training Launcher")
    parser = add_core_args(parser)
    args = parser.parse_args()
    args = validate_args(args, frozen_args)

    # Invoke the specified trainer
    trainer = get_trainer(known_args.trainer)(args)
    trainer.train()
    trainer.exit()


if __name__ == "__main__":
    train()
