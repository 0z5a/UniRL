# This module is copied from angelptm.megatron.training.arguments and modified to fit ar codebase.
# In the future, these modifications should be contributed back to AngelPTM.
#
import json
import re
from collections import defaultdict
from typing import List, Dict, Union, Any, Tuple, Optional


def normalize_arg_key(name: str) -> str:
    return name.replace("-", "_").lower()

def coerce_cli_scalar(val: str) -> Union[str, int, float, bool]:
    v = val.strip()
    low = v.lower()
    if low in ("true", "yes", "y", "on"):
        return True
    if low in ("false", "no", "n", "off"):
        return False
    if re.fullmatch(r"-?\d+", v):
        try:
            return int(v, base=10)
        except ValueError:
            pass
    try:
        return float(v)
    except ValueError:
        return val

def parse_argv_override_pairs(argv: List[str]) -> List[Tuple[str, Any]]:
    """Parse argparse-style tokens into (arg_name, value); mirrors common --k v / --k=v / --flag / --no-k forms."""
    pairs: List[Tuple[str, Any]] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        i += 1
        if not tok.startswith("--") or len(tok) < 3:
            continue
        rest = tok[2:]
        if rest.startswith("no-") and len(rest) > 3 and "=" not in rest:
            pairs.append((rest[3:], False))
            continue
        if "=" in rest:
            name, _, raw = rest.partition("=")
            pairs.append((name, coerce_cli_scalar(raw)))
            continue
        name = rest
        if i < len(argv) and not argv[i].startswith("--"):
            pairs.append((name, coerce_cli_scalar(argv[i])))
            i += 1
        else:
            pairs.append((name, True))
    return pairs

def leaf_paths_by_arg_name(config_field, prefix: str = "") -> Dict[str, List[str]]:
    """Map normalized YAML leaf key -> dotted paths for nodes that become --<key> argv (same rules as flatten)."""
    from omegaconf.dictconfig import DictConfig
    from omegaconf.listconfig import ListConfig

    index: Dict[str, List[str]] = defaultdict(list)

    def walk(cfg, pfx: str) -> None:
        for key, value in cfg.items():
            full_key = f"{pfx}.{key}" if pfx else key
            if full_key.startswith("__GLOBAL_VARS__"):
                continue
            if "__FROZEN__" in full_key:
                continue
            if isinstance(value, DictConfig):
                walk(value, full_key)
            elif isinstance(value, ListConfig):
                continue
            elif isinstance(value, bool):
                index[normalize_arg_key(key)].append(full_key)
            else:
                index[normalize_arg_key(key)].append(full_key)

    walk(config_field, prefix)
    return index

def apply_argv_overrides_to_omegaconf(cfg, argv_overrides: List[str]) -> None:
    """Merge CLI overrides into the loaded config before flattening so ${...} interpolations re-resolve."""
    from omegaconf import OmegaConf

    if not argv_overrides:
        return
    pairs = parse_argv_override_pairs(argv_overrides)
    if not pairs:
        return
    leaf_index = leaf_paths_by_arg_name(cfg)
    OmegaConf.set_struct(cfg, False)
    for name, val in pairs:
        paths = leaf_index.get(normalize_arg_key(name), [])
        for path in paths:
            OmegaConf.update(cfg, path, val, merge=True)

def parse_argv_from_yaml(
        yaml_path: str,
        allow_frozen: bool = False,
        argv_overrides: Optional[List[str]] = None,
) -> Union[List[str], Tuple[List[str], Dict[str, Any]]]:
    """Parse a YAML configuration file and convert it into a list of command-line arguments.

    NOTE: Write PTM args in YAML format:
    0. all arguments specified here are one-one correspondences with PTM arguments; un-specified arguments use PTM default values.
    1. arguments like '--hidden-size=2560' are written as 'hidden-size: 2560' (see below, note the space after colon).
    2. YAML format is like a tree, and only the leaf nodes are considered as arguments.
       Non-leaf nodes and structures are ignored, so arguments can be grouped as you like.
    3. arguments like '--sequence-parallel' can be written as 'enabled: \n  - sequence-parallel' (see below, note the indent).
       multiple `enabled` arguments in a same sub-tree must be put in a `enabled` list (see below for an example).
    4. If some nodes starting with `__FROZEN__` , the first level keys will be treated as leaf nodes even if
       they have sub-nodes. These leaf nodes will be directly added to args without being handled by argparse,
       therefore they also will not be added to argv. It is useful in several cases, e.g., one don't want too many
       configs to be flattened, or some arguments with the same name but in different subtrees.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        allow_frozen (bool, optional): Whether to allow frozen arguments. Frozen arguments are those
            under the `__FROZEN__*` node. Defaults to False.
        argv_overrides (list[str] | None): Optional argv fragment (e.g. ``parse_known_args`` remainder).
            When set, overrides are merged into the YAML :class:`~omegaconf.DictConfig` **before** it is
            flattened to ``argv``, so values that reference overridden keys via ``${...}`` stay
            consistent. Keys only present in ``argv_overrides`` and not in the YAML are ignored here
            and still take effect when the returned ``argv`` is concatenated with the same fragment
            for final argparse parsing.

    Returns:
        List[str]: A list of command-line arguments derived from the YAML configuration.
    """
    from omegaconf import OmegaConf
    from omegaconf.dictconfig import DictConfig
    from omegaconf.listconfig import ListConfig
    OmegaConf.register_new_resolver("add", lambda *args: sum(int(x) for x in args), replace=True)
    OmegaConf.register_new_resolver("min", lambda *args: min(int(x) for x in args), replace=True)
    OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y), replace=True)
    OmegaConf.register_new_resolver("div", lambda x, y: int(x) // int(y), replace=True)

    config_data = OmegaConf.load(yaml_path)
    if argv_overrides:
        apply_argv_overrides_to_omegaconf(config_data, argv_overrides)
    argv = []
    frozen_args = {}

    def _flatten_config_to_args(
        config_field: Union[Dict[str, Any], DictConfig], prefix: str = ""
    ):
        for key, value in config_field.items():
            full_key = f"{prefix}.{key}" if prefix != "" else key

            if full_key.startswith("__GLOBAL_VARS__"):
                continue

            if "__FROZEN__" in full_key:
                if not isinstance(value, DictConfig):
                    raise ValueError(f"__FROZEN__* node must be a DictConfig, got {type(value)}")
                for frozen_key, frozen_value in value.items():
                    if isinstance(frozen_value, (DictConfig, ListConfig)):
                        frozen_args[frozen_key] = OmegaConf.to_container(frozen_value, resolve=True)
                    else:
                        frozen_args[frozen_key] = frozen_value
                continue

            if isinstance(value, DictConfig):
                _flatten_config_to_args(value, prefix=full_key)
            elif isinstance(value, ListConfig):
                if key == "enabled":
                    for item in value:
                        argv.append(f"--{item}")
                else:
                    argv.append(f"--{key}")
                    for item in value:
                        if isinstance(item, DictConfig):
                            # Dict behind a list can not be handled by argument parser,
                            # so we serialize it as a JSON string.
                            resolved_dict = OmegaConf.to_container(item, resolve=True)
                            argv.append(json.dumps(resolved_dict))
                        else:
                            assert item is not None, f"null is not allowed in yaml config."
                            argv.append(str(item))
            elif isinstance(value, bool):
                if value:
                    argv.append(f"--{key}")
                else:
                    argv.append(f"--no-{key}")
            else:
                real_value = str(OmegaConf.select(config_data, full_key))
                assert real_value != "None", f"null is not allowed in yaml config for key: {full_key}"
                argv.append(f"--{key}")
                argv.append(real_value)

    _flatten_config_to_args(config_data)

    if allow_frozen:
        return argv, frozen_args
    return argv
