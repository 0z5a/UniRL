import json
import re
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Mapping, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import loguru
from loguru import logger
from natsort import natsorted
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache
from transformers.modeling_utils import load_state_dict as hf_load
from transformers.utils.generic import ModelOutput
try:
    from hy_parallelism.checkpoint.stateful import ModelWrapper
except ImportError:
    ModelWrapper = object

from hymm.core.parallel_states import ParallelState
from hymm.core.global_vars import get_args, get_parallel_state
from hymm.models.basic.model_config import TransformerConfig

HF_CONFIG_NAME = "config.json"
HF_WEIGHT_INDEX_NAME = "model.safetensors.index.json"

DCP_METADATA_NAME = ".metadata"

BIN_CONFIG_NAME = "config.json"
BIN_WEIGHT_INDEX_NAME = "pytorch_model.bin.index.json"

ALLOWED_MISMATCHED_KEYS = {
    # Add any keys that are allowed to have shape mismatches here
    "model.embed_tokens.weight",
    "lm_head.weight"
}
INPUT_EMBEDDINGS_KEY = "model.embed_tokens.weight"
OUTPUT_EMBEDDINGS_KEY = "lm_head.weight"


def align_tensor_shape(name, load_tensor, model_tensor):
    if load_tensor.shape == model_tensor.shape:
        return load_tensor
    if name not in ALLOWED_MISMATCHED_KEYS:
        raise ValueError(
            f"Shape mismatch for parameter {name}: expected {model_tensor.shape}, "
            f"got {load_tensor.shape}"
        )
    # Implement any specific shape alignment logic here if needed
    if name in ["model.embed_tokens.weight", "lm_head.weight"]:
        # If the vocab size is different, we can resize the embedding matrix
        assert len(load_tensor.shape) == 2 and len(model_tensor.shape) == 2, \
            f"Expected 2D tensors for {name}, got {load_tensor.shape} and {model_tensor.shape}"
        assert load_tensor.shape[1] == model_tensor.shape[1], \
            f"{name} dimension mismatch for {name}: expected {model_tensor.shape[1]}, got {load_tensor.shape[1]}"
        assert load_tensor.shape[0] <= model_tensor.shape[0], \
            f"Cannot align {name}: loaded tensor size {load_tensor.shape[0]} " \
            f"is larger than model tensor size {model_tensor.shape[0]}"
        # Clone the model tensor as new tensor and copy the loaded values
        if isinstance(model_tensor, DTensor):
            model_tensor = model_tensor.full_tensor()
        new_tensor = model_tensor.data.clone()
        new_tensor[:load_tensor.shape[0], :] = load_tensor
        return new_tensor

    else:
        raise NotImplementedError(f"Shape alignment not implemented for parameter {name}")


def _find_missing_and_unexpected_keys(
        model: nn.Module,
        checkpoint_keys: list[str],
) -> tuple[list[str], list[str]]:
    """ A simplified version of PreTrainedModel._find_missing_keys_and_unexpected_keys """

    # Compute expected keys, i.e. keys that the FULL model (not model_to_load) expects
    expected_keys = list(model.state_dict().keys())

    # Adjust prefix of the keys to make them match loaded keys before removing them
    missing_keys = sorted(set(expected_keys) - set(checkpoint_keys))
    unexpected_keys = set(checkpoint_keys) - set(expected_keys)

    # Remove nonpersistent buffers from unexpected keys: they are not in the expected keys (model state dict), but
    # may be in the loaded keys. Note that removing all buffers does the job, as they were part of the expected keys anyway
    model_buffers = {n for n, _ in model.named_buffers()}
    unexpected_keys = sorted(unexpected_keys - model_buffers)

    return missing_keys, unexpected_keys


def _find_mismatched_keys_and_get_key_file_mapping(
        model: nn.Module,
        checkpoint_files: Optional[list[str | Path]],
        keys_to_rename_mapping: dict[str, str],
        allowed_mismatched_keys: Optional[set[str]] = None,
) -> tuple[list[str], list[tuple[int, int]], dict[str, str]]:
    """ A simplified version of PreTrainedModel._find_mismatched_keys """

    # An error will be raised later on anyway if there is a mismatch - this avoids running the rest of this function
    # if there are no mismatch (which is almost always the case)

    model_state_dict = model.state_dict()
    mismatched_keys = []
    mismatched_shapes = []
    allowed_mismatched_keys = allowed_mismatched_keys or set()
    key_to_shard_file = {}

    for shard_file in checkpoint_files:
        state_dict = hf_load(str(shard_file), map_location="meta", weights_only=True)

        # Fix the key names
        new_state_dict = {keys_to_rename_mapping[k]: v for k, v in state_dict.items() if k in keys_to_rename_mapping}

        for key in new_state_dict.keys():
            key_to_shard_file[key] = shard_file
            if key in model_state_dict and new_state_dict[key].shape != model_state_dict[key].shape \
                    and key not in allowed_mismatched_keys:
                mismatched_keys.append(key)
                mismatched_shapes.append((new_state_dict[key].shape, model_state_dict[key].shape))

    return mismatched_keys, mismatched_shapes, key_to_shard_file


class LazyStateDict(Mapping):
    """
    A wrapper for a state dict that loads weights lazily.
    The actual loading is deferred until the state dict is accessed.
    """
    def __init__(self, state_dict=None, submodule_names=None):
        if state_dict is None:
            self._state_dict = {}
        else:
            assert isinstance(state_dict, dict), f'state_dict must be a dict, got {type(state_dict)}'
            self._state_dict = state_dict

        self.submodule_names = submodule_names

    def __len__(self):
        return len(self._state_dict)

    def __getitem__(self, key):
        # load the weight if not loaded yet
        value = self._state_dict[key]
        if isinstance(value, torch.Tensor):
            return value
        elif callable(value):
            value(key, self._state_dict)
        else:
            raise ValueError(f'Invalid value type for key {key}: {type(value)}')

        value = self._state_dict[key]
        assert isinstance(value, torch.Tensor), \
            f'Value for key {key} should be a torch.Tensor after loading, got {type(value)}'
        return value

    def __contains__(self, key):
        return key in self._state_dict

    def __iter__(self):
        return iter(self._state_dict)

    def keys(self):
        yield from self._state_dict.keys()

    def values(self):
        for key in self._state_dict:
            yield self[key]

    def items(self):
        for key in self._state_dict:
            yield key, self[key]

    def pop(self, key):
        value = self[key]
        del self._state_dict[key]
        return value

    def lazy_update(self, other):
        # unwrap to avoid invoking __getitem__ of other LazyStateDict
        if isinstance(other, LazyStateDict):
            other = other._state_dict
        for key in other.keys():
            self._state_dict[key] = other[key]

    def clear(self):
        self._state_dict.clear()

    def find_diff(self, model, ignore_unexpected_keys=None):
        # Find missing and unexpected keys from the state dict
        if model is not None:
            checkpoint_keys = list(self._state_dict.keys())
            missing_keys, unexpected_keys = _find_missing_and_unexpected_keys(model, checkpoint_keys)

            if ignore_unexpected_keys is not None:
                ignore_unexpected_keys = [re.compile(prefix) for prefix in ignore_unexpected_keys]
                unexpected_keys = [
                    k for k in unexpected_keys
                    if not any(prefix.match(k) for prefix in ignore_unexpected_keys)
                ]
        else:
            missing_keys, unexpected_keys = None, None
        return missing_keys, unexpected_keys


@dataclass
class LoadPlan:
    source: str
    state_dict: Optional[LazyStateDict]
    name: str
    metadata: Optional[dict[str, Any]] = None


def _is_hf_checkpoint_dir(weight_path: Path) -> bool:
    hf_index_file = weight_path / HF_WEIGHT_INDEX_NAME
    return hf_index_file.exists()


def _is_dcp_checkpoint_dir(weight_path: Path) -> bool:
    dcp_metadata_file = weight_path / DCP_METADATA_NAME
    return dcp_metadata_file.exists()


def _is_bin_checkpoint_dir(weight_path: Path) -> bool:
    bin_index_file = weight_path / BIN_WEIGHT_INDEX_NAME
    return bin_index_file.exists()


class HunyuanMultimodalState(nn.Module):
    """
    This class is a utility to handle the HunyuanMultimodal model weights loading from various sources.
    Supported sources include:
    - Submodules: submodules implementing get_pretrained_state_dict method.
    - Pure Torch state dict: a weight directory with __{rank}_{file_count}.distcp files.
    - HuggingFace checkpoint: a weight directory with model-{num}-of-{total}.safetensors files.
    """
    _config: TransformerConfig
    args: Namespace
    before_fsdp_plans: list[LoadPlan]
    fsdp_plans: list[LoadPlan]
    after_fsdp_plans: list[LoadPlan]

    base_model_prefix: str = ""

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_input_embeddings_key(self):     # noqa
        return INPUT_EMBEDDINGS_KEY

    def get_output_embeddings(self):
        return getattr(self, "lm_head", None)

    def get_output_embeddings_key(self):    # noqa
        return OUTPUT_EMBEDDINGS_KEY

    @staticmethod
    def has_valid_checkpoints(ckpt_dir: str | Path | None):
        if ckpt_dir is None or not Path(ckpt_dir).exists():
            return False
        for subdir in Path(ckpt_dir).iterdir():
            if subdir.is_dir() and subdir.name.startswith("iter_"):
                return True
        return False

    def collect_load_plans(
            self,
            checkpoint_dir: Optional[str | Path] = None,
            load_dir: Optional[str | Path] = None,
            fuse_experts_in_load: bool = False,
            copy_mot_in_load: bool = False,
    ):
        """
        Collect load plans from various sources.

        Args:
            checkpoint_dir:
            load_dir:
            fuse_experts_in_load (bool): Whether to use fused experts weights while loading checkpoints.

        """
        args = self.args
        load_optimizer_states = False if getattr(args, 'no_load_optim', False) else None
        # We have multiple weights loading sources, resulting complexity in loading plans.
        # Therefore, we define the loading plans here to keep track of the loading process.
        self.before_fsdp_plans = []
        self.fsdp_plans = []
        self.after_fsdp_plans = []

        # args.resume loading plan
        if args.resume:
            latest_tag = None

            # If checkpoint_dir is empty while load_dir is a standard checkpoint directory, we can also try to
            # resume from load_dir for adapting the schema of Taiji Platform.
            if not self.has_valid_checkpoints(checkpoint_dir) and self.has_valid_checkpoints(load_dir):
                checkpoint_dir = Path(load_dir)

            if checkpoint_dir is not None and checkpoint_dir.exists():
                # Find latest checkpoint matching `iter_{:07d}` pattern.
                for subdir in checkpoint_dir.iterdir():
                    if subdir.is_dir() and subdir.name.startswith("iter_"):
                        if latest_tag is None or subdir.name > latest_tag:
                            latest_tag = subdir.name
                if latest_tag is not None:
                    plan = LoadPlan(
                        source="resume", state_dict=None, name=str(checkpoint_dir), metadata=dict(
                            load_dir=checkpoint_dir, tag=latest_tag,
                            load_optimizer_states=load_optimizer_states,
                        )
                    )
                    self.after_fsdp_plans.append(plan)
                    # If resume is enabled and find a valid checkpoint, we assume no other loading is needed.
                    if args.load_pretrained_submodules or load_dir:
                        print(f"[rank {dist.get_rank()}] Load planing: "
                              f"Resume checkpoint found at {checkpoint_dir}/{latest_tag}, "
                              f"args.load_pretrained_submodules({args.load_pretrained_submodules}) "
                              f"and load_dir({load_dir}) will be ignored.")
                    return

        # If use meta device, to_empty is always the first step to ensure all parameters are initialized.
        use_meta_device = args.init_device == "meta"

        # Pretrained submodules loading plan. For those modules implementing `get_pretrained_state_dict` method.
        # If the submodule is too large to load immediately, avoid implementing this method, and handle loading
        # in FSDP initialization.
        if args.load_pretrained_submodules:
            state_dict_from_submodules = self.state_dict_from_pretrained_submodules()
            if len(state_dict_from_submodules) > 0:
                plan = LoadPlan(
                    source="submodules", state_dict=state_dict_from_submodules, name="vit"
                )
                if use_meta_device:
                    self.fsdp_plans.append(plan)
                else:
                    self.before_fsdp_plans.append(plan)

        # pretrained model loading plan.
        if load_dir is not None:
            weight_path = self._check_is_directory(load_dir)

            # Determine if the weights are in HuggingFace format (safetensors)
            if _is_hf_checkpoint_dir(weight_path):
                state_dict_from_hf = self.state_dict_from_hf(
                    weight_path, fuse_experts_in_load=fuse_experts_in_load, copy_mot_in_load=copy_mot_in_load
                )
                if len(state_dict_from_hf) == 0:
                    raise ValueError(f'No weights found in HuggingFace checkpoint at {weight_path}.')
                plan = LoadPlan(
                    source="huggingface", state_dict=state_dict_from_hf, name=str(weight_path)
                )
                if use_meta_device:
                    self.fsdp_plans.append(plan)
                else:
                    self.before_fsdp_plans.append(plan)

            # Determine if the weights are in torch format (pytorch_model*.bin)
            elif _is_bin_checkpoint_dir(weight_path):
                state_dict_from_bin = self.state_dict_from_bin(weight_path, fuse_experts_in_load=fuse_experts_in_load)
                if len(state_dict_from_bin) == 0:
                    raise ValueError(f'No weights found in bin checkpoint at {weight_path}.')
                plan = LoadPlan(
                    source="pytorch_bin", state_dict=state_dict_from_bin, name=str(weight_path)
                )
                if use_meta_device:
                    self.fsdp_plans.append(plan)
                else:
                    self.before_fsdp_plans.append(plan)

            # Determine if the weights are in DCP format
            elif _is_dcp_checkpoint_dir(weight_path):
                state_dict_from_dcp = self.state_dict_from_dcp(weight_path)
                if len(state_dict_from_dcp) == 0:
                    raise ValueError(f'No weights found in DCP checkpoint at {weight_path}.')
                plan = LoadPlan(
                    source="dcp", state_dict=state_dict_from_dcp, name=str(weight_path),
                    metadata=dict(
                        load_dir=weight_path, load_optimizer_states=False, load_lr_scheduler_states=False,
                    )
                )
                self.after_fsdp_plans.append(plan)

            # If not matched implementation, raise error
            else:
                raise ValueError(
                    f'Weight directory {weight_path} does not contain a valid checkpoint format: '
                    f'huggingface, dcp, or pytorch bin.'
                )

    def load_before_fsdp(self):
        """
        Execute the loading plans in before_fsdp_plans.
        These weight initialization plans will be immediately executed.
        """
        if len(self.before_fsdp_plans) == 0:
            return

        # Merge all before_fsdp_plans into a single lazy_state_dict for loading
        lazy_state_dict = LazyStateDict()
        sources = []

        for plan in self.before_fsdp_plans:
            lazy_state_dict.lazy_update(plan.state_dict)
            sources.append(f"{plan.source}({plan.name})")

        # Load the weights from lazy_state_dict
        missing_keys, unexpected_keys = self.load_state_dict(lazy_state_dict, strict=False)

        rank = torch.distributed.get_rank()
        print(f"[rank {rank}] Missing keys: {missing_keys}\n"
              f"[rank {rank}] Unexpected keys: {unexpected_keys}\n"
              f"[rank {rank}] (ignored unexpected pattern: {self.args.ignore_unexpected_keys})",
              flush=True)
        print(f"Collected {len(lazy_state_dict)} parameters in lazy_state_dict for stream loading "
              f"from following sources: {sources}", flush=True)
        # Wait for all ranks to finish loading
        torch.distributed.barrier()

    def state_dict_from_pretrained_submodules(self) -> LazyStateDict:
        """
        Get state dict from submodules implementing get_pretrained_state_dict method.
        Suitable for small models or small submodules where loading weights immediately is feasible.
        """
        state_dict = {}
        submodule_names = []
        for name, module in self.named_modules():
            if hasattr(module, 'get_pretrained_state_dict'):
                submodule_state_dict = module.get_pretrained_state_dict()
                for k, v in submodule_state_dict.items():
                    state_dict[f'{name}.{k}'] = v
                submodule_names.append(name)
        return LazyStateDict(state_dict, submodule_names=submodule_names)

    @staticmethod
    def _apply_key_mapping(key, key_mapping):
        new_key, has_changed = key, False
        # Optionally map the key according to `key_mapping`
        for patterns, replacements in key_mapping.items():
            if isinstance(patterns, str):
                patterns = (patterns,)
                replacements = (replacements,)
            group_n_replace = 0
            for pattern, replacement in zip(patterns, replacements):
                if '>' in pattern:
                    prefix, pattern = pattern.split('>')
                    if not key.startswith(prefix):
                        continue
                new_key, n_replace = re.subn(pattern, replacement, new_key)
                group_n_replace += n_replace
            # Early exit of the loop
            if group_n_replace > 0:
                has_changed = True
                break
        return new_key, has_changed

    def get_checkpoint_key_renaming_mapping(
            self,
            checkpoint_keys: list[str],
            key_mapping: Optional[dict[str, str]] = None,
            loading_base_model_from_task_state_dict: bool = False,
            loading_task_model_from_base_state_dict: bool = False,
    ):
        """
        Compute a mapping between the serialized keys on disk `checkpoint_keys`, and the keys that the model
        that we are loading expects. This is the single entry point for key renaming that will be used during
        loading.
        Log if any parameters have been renamed.
        """
        prefix = self.base_model_prefix
        _prefix = f"{prefix}."

        if key_mapping is None:
            key_mapping = self.get_key_mapping()

        key_renaming_mapping = {}

        for key in checkpoint_keys:
            new_key, has_changed = self._apply_key_mapping(key, key_mapping)

            # In this case, we need to add the prefix to the keys, to match them to the expected keys
            if loading_task_model_from_base_state_dict:
                new_key = ".".join([prefix, new_key])
            # In this case we need to remove the prefix from the key to match them to the expected keys, and use
            # only the keys starting with the prefix
            elif loading_base_model_from_task_state_dict:
                if not new_key.startswith(_prefix):
                    continue
                new_key = new_key[len(_prefix):]

            _ = has_changed
            # Currently `has_changed` is not used.
            key_renaming_mapping[key] = new_key

        return key_renaming_mapping

    def _get_key_renaming_mapping(self, *args, **kwargs):
        return self.get_checkpoint_key_renaming_mapping(*args, **kwargs)

    def get_model_key_renaming_mapping(
            self,
            model_param_keys: Optional[Iterable[str]] = None,
            key_mapping: Optional[dict[str, str]] = None,
    ):
        key_renaming_mapping = {}
        if model_param_keys is None:
            model_param_keys = list([name for name, _ in self.named_parameters()])
        if key_mapping is None:
            key_mapping = self.get_key_mapping()

        for key in model_param_keys:
            new_key, has_changed = self._apply_key_mapping(key, key_mapping)
            _ = has_changed
            # Currently `has_changed` is not used.
            key_renaming_mapping[key] = new_key

        return key_renaming_mapping, key_mapping

    def _get_model_tensor(self, key: str):
        from hy_parallelism.distributed.fsdp_util import get_fsdp_named_parameters
        for name, param in get_fsdp_named_parameters(self):
            if name == key:
                return param
        return None

    def state_dict_from_hf(
            self,
            weight_path: str | Path,
            index_name: Optional[str] = None,
            key_mapping: Optional[dict[str, str]] = None,
            fuse_experts_in_load: bool = False,
            copy_mot_in_load: bool = False,
    ) -> LazyStateDict:
        """
        Get state dict from HuggingFace checkpoint files.
        Suitable for large models whose weights are sharded across multiple files.
        """
        weight_path = Path(weight_path)
        if not weight_path.exists() or not weight_path.is_dir():
            raise FileNotFoundError(f'Weight path {weight_path} does not exist or is not a directory.')

        # Load config.json to get model architecture information
        config_file = weight_path / HF_CONFIG_NAME
        if not config_file.exists():
            raise FileNotFoundError(f'HuggingFace config file {HF_CONFIG_NAME} not found in {weight_path}.')
        with config_file.open() as f:
            config_data = json.load(f)
        self._tie_word_embeddings = config_data.get("tie_word_embeddings", False)

        # Load index file for collecting weight keys
        if index_name is None:
            index_name = HF_WEIGHT_INDEX_NAME
        index_file = weight_path / index_name
        if not index_file.exists():
            raise FileNotFoundError(f'HuggingFace index file {HF_WEIGHT_INDEX_NAME} not found in {weight_path}.')
        with index_file.open() as f:
            index_data = json.load(f)
        if "weight_map" not in index_data:
            raise KeyError(f'weight_map not found in HuggingFace index file {index_file}.')

        # Define key mapping for renaming keys if provided
        original_checkpoint_keys = list(index_data["weight_map"].keys())
        key_renaming_mapping = self.get_checkpoint_key_renaming_mapping(
            original_checkpoint_keys,
            key_mapping,
        )
        checkpoint_keys = list(key_renaming_mapping.values())

        # Assign tied weights if applicable
        if self._tie_word_embeddings:
            input_embeddings_key = self.get_input_embeddings_key()
            output_embeddings_key = self.get_output_embeddings_key()
            assert input_embeddings_key in checkpoint_keys and output_embeddings_key not in checkpoint_keys, \
                (f"Tied weights require {input_embeddings_key} to be present "
                 f"and {output_embeddings_key} to be absent in checkpoint.")
            key_renaming_mapping[output_embeddings_key] = output_embeddings_key
            checkpoint_keys.append(output_embeddings_key)

        # Append fused expert keys if applicable
        moe_impl = getattr(self._config, "moe_impl", "hunyuan")
        if fuse_experts_in_load:
            assert self._config.split_gate_and_up, \
                "Fused experts loading requires split_gate_and_up to be True."
            fused_expert_keys = []
            processed_layer = set()
            pattern = re.compile(r"(.+)\.\d+\.gate_proj\.weight")
            for key in checkpoint_keys:
                if (match := pattern.match(key)) is not None:
                    layer_prefix = match.group(1)
                    if layer_prefix in processed_layer:
                        continue
                    processed_layer.add(layer_prefix)
                    for subname in ["gate", "up", "down"]:
                        fused_key = f"{layer_prefix}.{subname}_proj_weights"
                        fused_expert_keys.append(fused_key)
                        key_renaming_mapping[fused_key] = fused_key
            checkpoint_keys.extend(fused_expert_keys)

        # FlashInferMoE at load time has per-expert params, but checkpoint may have fused keys (three _proj_weights).
        # Detect and replace: remove fused keys, add per-expert keys, and unfuse in load_fn.
        ckpt_has_fused_experts = any(
            re.search(r"\.experts\.(gate_proj_weights|up_proj_weights|down_proj_weights)$", k)
            for k in checkpoint_keys
        )
        do_unfuse_for_flashinfer = (moe_impl == "flashinfer") and ckpt_has_fused_experts
        if do_unfuse_for_flashinfer:
            num_experts = self._config.num_experts
            fused_to_remove = []
            per_expert_to_add = []
            processed_layer = set()
            three_fused_pattern = re.compile(r"(.+)\.(?:gate|up|down)_proj_weights$")
            for key in checkpoint_keys:
                match = three_fused_pattern.match(key)
                if match is not None:
                    layer_prefix = match.group(1)  # e.g. model.layers.1.mlp.experts
                    if layer_prefix in processed_layer:
                        continue
                    processed_layer.add(layer_prefix)
                    for fused_name in ["gate_proj_weights", "up_proj_weights", "down_proj_weights"]:
                        fused_to_remove.append(f"{layer_prefix}.{fused_name}")
                    for i in range(num_experts):
                        for proj in ["gate_proj", "up_proj", "down_proj"]:
                            per_expert_key = f"{layer_prefix}.{i}.{proj}.weight"
                            per_expert_to_add.append(per_expert_key)
                            key_renaming_mapping[per_expert_key] = per_expert_key
            fused_to_remove_set = set(fused_to_remove)
            checkpoint_keys[:] = [k for k in checkpoint_keys if k not in fused_to_remove_set]
            checkpoint_keys.extend(per_expert_to_add)

        if copy_mot_in_load:
            copied_keys = []
            processed_layer = set()
            for key in checkpoint_keys:
                for pattern in [
                    '.qkv_proj', '.q_proj', '.k_proj', '.v_proj', '.o_proj',
                    '.query_layernorm', '.key_layernorm', 'input_layernorm', '.post_attention_layernorm', 
                    '.mlp'
                ]:
                    if pattern in key:
                        copied_key = key.replace(pattern, pattern + '_mot_gen')
                        copied_keys.append(copied_key)
            checkpoint_keys.extend(copied_keys)

        # After copy_mot_in_load may have re-introduced fused keys via .mlp→.mlp_mot_gen, remove them again.
        if do_unfuse_for_flashinfer:
            fused_pattern = re.compile(r"\.experts\.(gate_proj_weights|up_proj_weights|down_proj_weights)$")
            checkpoint_keys[:] = [k for k in checkpoint_keys if fused_pattern.search(k) is None]

        # Collect checkpoint files
        checkpoint_files = sorted(weight_path.glob('model-*-of-*.safetensors'))
        if not checkpoint_files:
            raise FileNotFoundError(f'No HuggingFace checkpoint files found in {weight_path}.')
        named_num_checkpoint_files = int(checkpoint_files[0].stem.split('-of-')[1])
        if len(checkpoint_files) != named_num_checkpoint_files:
            raise ValueError(
                f'Number of checkpoint files found ({len(checkpoint_files)}) does not match '
                f'the number indicated in the first file name ({named_num_checkpoint_files}).'
            )

        # Find all the keys with shape mismatch (if we ignore the mismatch, the weights need to be newly initialized
        # the same way as missing keys). When fuse_experts_in_load, allow fused expert keys to have different
        # expert dimension (e.g. ckpt [128,...] vs model [8,...]); we slice when loading.
        allowed_mismatched = set(ALLOWED_MISMATCHED_KEYS)
        if fuse_experts_in_load:
            for k in checkpoint_keys:
                if "experts" in k and (
                    "gate_proj_weights" in k or "up_proj_weights" in k or "down_proj_weights" in k
                ):
                    allowed_mismatched.add(k)
        if do_unfuse_for_flashinfer:
            for k in checkpoint_keys:
                if re.search(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight", k):
                    allowed_mismatched.add(k)
        mismatched_keys, mismatched_shapes, key_to_shard_file = \
            _find_mismatched_keys_and_get_key_file_mapping(
                self, checkpoint_files, key_renaming_mapping, allowed_mismatched_keys=allowed_mismatched,
            )
        if len(mismatched_keys) > 0:
            error_info = "\n".join(
                [f"Key: {k}, checkpoint shape: {cs}, model shape: {ms}"
                 for k, (cs, ms) in zip(mismatched_keys, mismatched_shapes)]
            )
            raise ValueError(f"Found {len(mismatched_keys)} mismatched keys:\n{error_info}")
        if self._tie_word_embeddings:
            input_embeddings_key = self.get_input_embeddings_key()
            output_embeddings_key = self.get_output_embeddings_key()
            key_to_shard_file[output_embeddings_key] = key_to_shard_file[input_embeddings_key]

        # Collect fused keys that exist in ckpt (for flashinfer unfuse lookup)
        _fused_ckpt_keys = set()
        if do_unfuse_for_flashinfer:
            fused_pat = re.compile(r"\.experts\.(gate_proj_weights|up_proj_weights|down_proj_weights)$")
            for orig_key in original_checkpoint_keys:
                renamed = key_renaming_mapping.get(orig_key, orig_key)
                if fused_pat.search(renamed):
                    _fused_ckpt_keys.add(renamed)

        def load_fn(key: str, state_dict: dict[str, Optional[torch.Tensor]]):
            # FlashInfer unfuse: ckpt has fused three _proj_weights → slice into per-expert keys
            if do_unfuse_for_flashinfer:
                m = re.match(r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", key)
                if m is not None:
                    experts_prefix = m.group(1)
                    expert_idx = int(m.group(2))
                    proj_name = m.group(3)  # gate_proj, up_proj, or down_proj
                    fused_key = f"{experts_prefix}.{proj_name}_weights"
                    # For _mot_gen keys, the fused key in ckpt is the non-_mot_gen version
                    if fused_key not in _fused_ckpt_keys and "_mot_gen" in fused_key:
                        src_fused_key = fused_key.replace("_mot_gen", "")
                    else:
                        src_fused_key = fused_key
                    if fused_key not in state_dict or not isinstance(state_dict.get(fused_key), torch.Tensor):
                        if src_fused_key in key_to_shard_file:
                            shard_file = key_to_shard_file[src_fused_key]
                            shard_state_dict = hf_load(str(shard_file), map_location="cpu", weights_only=True)
                            shard_state_dict = {key_renaming_mapping[k]: v for k, v in shard_state_dict.items()}
                            state_dict.update(shard_state_dict)
                        if fused_key not in state_dict and src_fused_key in state_dict:
                            state_dict[fused_key] = state_dict[src_fused_key]
                    fused_tensor = state_dict.get(fused_key)
                    if fused_tensor is not None and isinstance(fused_tensor, torch.Tensor):
                        state_dict[key] = fused_tensor[expert_idx].clone()
                        return
                    # If fused_tensor is not available, fall through to normal loading path

            # Handle fused experts loading if applicable (non-FlashInfer: gate/up/down_proj_weights)
            if fuse_experts_in_load and "experts" in key and (
                "gate_proj_weights" in key or "up_proj_weights" in key or "down_proj_weights" in key
            ):
                p_state: ParallelState = get_parallel_state()
                if p_state.ep_size > 1:
                    num_experts_per_ep_rank = self._config.num_experts // p_state.ep_size
                    expert_index_range = (
                        num_experts_per_ep_rank * p_state.ep_rank,
                        num_experts_per_ep_rank * (p_state.ep_rank + 1),
                    )
                else:
                    expert_index_range = (0, self._config.num_experts)
                subname = key.rsplit('.')[-1].split('_')[-3]
                subkey_lst = [
                    '.'.join(key.split('.')[:-1] + [str(expert_idx), f'{subname}_proj.weight'])
                    for expert_idx in range(*expert_index_range)
                ]
                tensors = []
                for subkey in subkey_lst:
                    if subkey not in state_dict or not isinstance(state_dict[subkey], torch.Tensor):
                        if copy_mot_in_load and 'mot_gen' in subkey:
                            continue
                        shard_file = key_to_shard_file[subkey]
                        shard_state_dict = hf_load(str(shard_file), map_location="cpu", weights_only=True)
                        shard_state_dict = {key_renaming_mapping[k]: v for k, v in shard_state_dict.items()}
                        state_dict.update(shard_state_dict)


                        for k, v in shard_state_dict.items():
                            for pattern in [
                                '.qkv_proj', '.q_proj', '.k_proj', '.v_proj', '.o_proj',
                                '.query_layernorm', '.key_layernorm', 'input_layernorm', '.post_attention_layernorm', 
                                '.mlp'
                            ]:
                                if pattern in k:
                                    copied_k = k.replace(pattern, pattern + '_mot_gen')
                                    if copied_k in checkpoint_keys:
                                        state_dict[copied_k] = state_dict[k].clone()

                    # Pop to avoid being treated unexpected keys
                    tensors.append(state_dict.pop(subkey))
                if len(tensors) == 0:
                    return
                state_dict[key] = torch.stack(tensors, dim=0)

                if copy_mot_in_load:
                    for pattern in [
                        '.qkv_proj', '.q_proj', '.k_proj', '.v_proj', '.o_proj',
                        '.query_layernorm', '.key_layernorm', 'input_layernorm', '.post_attention_layernorm', 
                        '.mlp'
                    ]:
                        if pattern in key:
                            copied_k = key.replace(pattern, pattern + '_mot_gen')
                            if copied_k in checkpoint_keys:
                                state_dict[copied_k] = state_dict[key].clone()

                # print(f"[rank {p_state.dp_rank}] Fused expert weight loaded for key {key} "
                #       f"with shape {state_dict[key].shape}")
                return

            # Note: This function maybe called during iterating over state_dict,
            # thus don't change the state_dict size here.
            if key not in state_dict:
                raise KeyError(f'Key {key} not found in state_dict.')

            # Load the shard file containing the key
            shard_file = key_to_shard_file[key]
            shard_state_dict = hf_load(str(shard_file), map_location="cpu", weights_only=True)
            key_value = None
            for k, v in shard_state_dict.items():
                if k in key_renaming_mapping:
                    new_k = key_renaming_mapping[k]
                else:
                    new_k = k
                if new_k in ALLOWED_MISMATCHED_KEYS:
                    model_tensor = self._get_model_tensor(new_k)
                    # If model_tensor is DTensor, the shape alignment will be handled in FSDP initialization.
                    if isinstance(model_tensor, torch.Tensor) and model_tensor.device != torch.device('meta'):
                        v = align_tensor_shape(new_k, v, model_tensor)
                # When fuse_experts_in_load, checkpoint may store fused experts with more experts (e.g. 128)
                # than the model (e.g. 8 per EP rank); slice to this rank's expert range.
                elif (
                    fuse_experts_in_load
                    and "experts" in new_k
                    and ("gate_proj_weights" in new_k or "up_proj_weights" in new_k or "down_proj_weights" in new_k)
                    and v.ndim >= 1
                ):
                    model_tensor = self._get_model_tensor(new_k)
                    if model_tensor is not None and v.shape[0] != model_tensor.shape[0]:
                        p_state: ParallelState = get_parallel_state()
                        if p_state.ep_size > 1:
                            num_experts_per_ep_rank = self._config.num_experts // p_state.ep_size
                            start = num_experts_per_ep_rank * p_state.ep_rank
                            end = num_experts_per_ep_rank * (p_state.ep_rank + 1)
                        else:
                            start, end = 0, model_tensor.shape[0]
                        v = v[start:end].contiguous()
                state_dict[new_k] = v
                
                # Track the value for the requested key
                if new_k == key:
                    key_value = v

                if copy_mot_in_load:
                    for pattern in [
                        '.qkv_proj', '.q_proj', '.k_proj', '.v_proj', '.o_proj',
                        '.query_layernorm', '.key_layernorm', 'input_layernorm', '.post_attention_layernorm', 
                        '.mlp'
                    ]:
                        if pattern in new_k:
                            copied_k = new_k.replace(pattern, pattern + '_mot_gen')
                            if copied_k in checkpoint_keys:
                                state_dict[copied_k] = state_dict[new_k].clone()

            # Ensure the requested key is set (it should be set in the loop above, but double-check)
            if key_value is not None:
                state_dict[key] = key_value
            elif key not in state_dict or not isinstance(state_dict.get(key), torch.Tensor):
                # If key was not found, try to get it from state_dict (it might have been set by another key)
                if key in state_dict and isinstance(state_dict[key], torch.Tensor):
                    pass  # Already set
                else:
                    raise KeyError(f'Key {key} not found in shard file {shard_file} after renaming.')

            # Assign tied weights if applicable
            if self._tie_word_embeddings:
                input_embeddings_key = self.get_input_embeddings_key()
                output_embeddings_key = self.get_output_embeddings_key()
                if isinstance(state_dict.get(input_embeddings_key, None), torch.Tensor) and not isinstance(
                        state_dict.get(output_embeddings_key, None), torch.Tensor):
                    state_dict[output_embeddings_key] = state_dict[input_embeddings_key].clone()
                    logger.debug(f"[rank {dist.get_rank()}] "
                                 f"Tied weight assigned: {output_embeddings_key} from {input_embeddings_key}",
                                 flush=True)

        state_dict = {k: load_fn for k in checkpoint_keys}
        return LazyStateDict(state_dict)

    def state_dict_from_dcp(self, weight_path: str | Path) -> LazyStateDict:
        """
        Get state dict from Distributed Checkpoint(DCP) files.
        Suitable for large models whose weights are sharded across multiple files.
        DCP load is highly integrated within torch, so the implementation here is simplified to just reading metadata.
        The real loading will be done after FSDP initialization.
        """
        _ = self
        weight_path = Path(weight_path)
        if not weight_path.exists() or not weight_path.is_dir():
            raise FileNotFoundError(f'Weight path {weight_path} does not exist or is not a directory.')

        reader = FileSystemReader(weight_path)
        metadata = reader.read_metadata()

        checkpoint_keys = list(metadata.state_dict_metadata.keys())
        # make sure all keys startswith `model.`, which means the dcp checkpoint is saved by hy-parallelism
        # if not all(key.startswith('model.') for key in checkpoint_keys):
        #     raise ValueError(f"All keys in DCP checkpoint must start with 'model.'")

        state_dict = {k: None for k in checkpoint_keys}
        return LazyStateDict(state_dict)

    def state_dict_from_bin(
            self,
            weight_path: str | Path,
            index_name: Optional[str] = None,
            key_mapping: Optional[dict[str, str]] = None,
            fuse_experts_in_load: bool = False,
    ) -> LazyStateDict:
        """
        Get state dict from pytorch_model*.bin checkpoint files.
        Suitable for large models whose weights are sharded across multiple files.
        """
        weight_path = Path(weight_path)
        if not weight_path.exists() or not weight_path.is_dir():
            raise FileNotFoundError(f'Weight path {weight_path} does not exist or is not a directory.')

        # Load config.json to get model architecture information
        config_file = weight_path / BIN_CONFIG_NAME
        if not config_file.exists():
            raise FileNotFoundError(f'Pytorch model bin config file {BIN_CONFIG_NAME} not found in {weight_path}.')
        with config_file.open() as f:
            config_data = json.load(f)
        self._tie_word_embeddings = config_data.get("tie_word_embeddings", False)

        # Load index file for collecting weight keys
        if index_name is None:
            index_name = BIN_WEIGHT_INDEX_NAME
        index_file = weight_path / index_name
        if not index_file.exists():
            raise FileNotFoundError(f'Pytorch model bin index file {BIN_WEIGHT_INDEX_NAME} not found in {weight_path}.')
        with index_file.open() as f:
            index_data = json.load(f)
        if "weight_map" not in index_data:
            raise KeyError(f'weight_map not found in Pytorch model bin index file {index_file}.')

        # Define key mapping for renaming keys if provided
        original_checkpoint_keys = list(index_data["weight_map"].keys())
        key_renaming_mapping = self.get_checkpoint_key_renaming_mapping(
            original_checkpoint_keys,
            key_mapping,
        )
        checkpoint_keys = list(key_renaming_mapping.values())

        # Assign tied weights if applicable
        if self._tie_word_embeddings:
            input_embeddings_key = self.get_input_embeddings_key()
            output_embeddings_key = self.get_output_embeddings_key()
            assert input_embeddings_key in checkpoint_keys and output_embeddings_key not in checkpoint_keys, \
                (f"Tied weights require {input_embeddings_key} to be present "
                 f"and {output_embeddings_key} to be absent in checkpoint.")
            key_renaming_mapping[output_embeddings_key] = output_embeddings_key
            checkpoint_keys.append(output_embeddings_key)

        # Append fused expert keys if applicable
        if fuse_experts_in_load:
            assert not self._config.split_gate_and_up, \
                "Fused experts loading requires split_gate_and_up to be False."
            fused_expert_keys = []
            processed_layer = set()
            pattern = re.compile(r"(.+)\.\d+\.gate_and_up_proj\.weight")
            for key in checkpoint_keys:
                if (match := pattern.match(key)) is not None:
                    layer_prefix = match.group(1)
                    if layer_prefix in processed_layer:
                        continue
                    processed_layer.add(layer_prefix)
                    for subname in ["gate", "up", "down"]:
                        fused_key = f"{layer_prefix}.{subname}_proj_weights"
                        fused_expert_keys.append(fused_key)
                        key_renaming_mapping[fused_key] = fused_key
            checkpoint_keys.extend(fused_expert_keys)
            # print(f"[rank {dist.get_rank()}] Fused expert keys added for loading: {fused_expert_keys}", flush=True)

        key_to_shard_file = {
            key_renaming_mapping[key]: str(weight_path / shard_file_name)
            for key, shard_file_name in index_data["weight_map"].items()
        }

        # Collect checkpoint files
        checkpoint_files = sorted(weight_path.glob('pytorch_model-*-of-*.bin'))
        if not checkpoint_files:
            raise FileNotFoundError(f'No Pytorch model bin checkpoint files found in {weight_path}.')
        named_num_checkpoint_files = int(checkpoint_files[0].stem.split('-of-')[1])
        if len(checkpoint_files) != named_num_checkpoint_files:
            raise ValueError(
                f'Number of checkpoint files found ({len(checkpoint_files)}) does not match '
                f'the number indicated in the first file name ({named_num_checkpoint_files}).'
            )

        def load_fn(key: str, state_dict: dict[str, Optional[torch.Tensor]]):
            # Handle fused experts loading if applicable
            if fuse_experts_in_load and "experts" in key and (
                "gate_proj_weights" in key or "up_proj_weights" in key or "down_proj_weights" in key
            ):
                p_state: ParallelState = get_parallel_state()
                if p_state.ep_size > 1:
                    num_experts_per_ep_rank = self._config.num_experts // p_state.ep_size
                    expert_index_range = (
                        num_experts_per_ep_rank * p_state.ep_rank,
                        num_experts_per_ep_rank * (p_state.ep_rank + 1),
                    )
                else:
                    expert_index_range = (0, self._config.num_experts)
                subname = key.rsplit('.')[-1].split('_')[-3]

                if subname == "down":
                    subkey_lst = [
                        '.'.join(key.split('.')[:-1] + [str(expert_idx), f'{subname}_proj.weight'])
                        for expert_idx in range(*expert_index_range)
                    ]
                    tensors = []
                    for subkey in subkey_lst:
                        if subkey not in state_dict or not isinstance(state_dict[subkey], torch.Tensor):
                            shard_file = key_to_shard_file[subkey]
                            shard_state_dict = hf_load(str(shard_file), map_location="cpu", weights_only=True)
                            shard_state_dict = {key_renaming_mapping[k]: v for k, v in shard_state_dict.items()}
                            state_dict.update(shard_state_dict)
                        # Pop to avoid being treated unexpected keys
                        tensors.append(state_dict.pop(subkey))
                    state_dict[key] = torch.stack(tensors, dim=0)
                else:
                    subkey_lst = [
                        '.'.join(key.split('.')[:-1] + [str(expert_idx), 'gate_and_up_proj.weight'])
                        for expert_idx in range(*expert_index_range)
                    ]
                    tensors = []
                    for subkey in subkey_lst:
                        if subkey not in state_dict or not isinstance(state_dict[subkey], torch.Tensor):
                            shard_file = key_to_shard_file[subkey]
                            shard_state_dict = hf_load(str(shard_file), map_location="cpu", weights_only=True)
                            shard_state_dict = {key_renaming_mapping[k]: v for k, v in shard_state_dict.items()}
                            state_dict.update(shard_state_dict)
                        # Pop to avoid being treated unexpected keys
                        tensors.append(state_dict.pop(subkey))
                    state_dict[key] = torch.stack(tensors, dim=0)
                    up_proj, gate_proj = state_dict[key].chunk(2, dim=1)
                    if "gate_proj_weight" in key:
                        state_dict[key] = gate_proj
                        state_dict[key.replace('gate_proj_weight', 'up_proj_weight')] = up_proj
                    else:
                        state_dict[key] = up_proj
                        state_dict[key.replace('up_proj_weight', 'gate_proj_weight')] = gate_proj
                # print(f"[rank {p_state.dp_rank}] Fused expert weight loaded for key {key} "
                #       f"with shape {state_dict[key].shape}")
                return

            # Note: This function maybe called during iterating over state_dict,
            # thus don't change the state_dict size here.
            if key not in state_dict:
                raise KeyError(f'Key {key} not found in state_dict.')

            # Load the shard file containing the key
            shard_file = key_to_shard_file[key]
            shard_state_dict = torch.load(str(shard_file), map_location="cpu", weights_only=True)
            for k, v in shard_state_dict.items():
                if k in key_renaming_mapping:
                    new_k = key_renaming_mapping[k]
                else:
                    new_k = k
                if new_k in ALLOWED_MISMATCHED_KEYS:
                    model_tensor = self._get_model_tensor(new_k)
                    # If model_tensor is DTensor, the shape alignment will be handled in FSDP initialization.
                    if isinstance(model_tensor, torch.Tensor) and model_tensor.device != torch.device('meta'):
                        v = align_tensor_shape(new_k, v, model_tensor)
                elif (
                    fuse_experts_in_load
                    and "experts" in new_k
                    and ("gate_proj_weights" in new_k or "up_proj_weights" in new_k or "down_proj_weights" in new_k)
                    and v.ndim >= 1
                ):
                    model_tensor = self._get_model_tensor(new_k)
                    if model_tensor is not None and v.shape[0] != model_tensor.shape[0]:
                        p_state: ParallelState = get_parallel_state()
                        if p_state.ep_size > 1:
                            num_experts_per_ep_rank = self._config.num_experts // p_state.ep_size
                            start = num_experts_per_ep_rank * p_state.ep_rank
                            end = num_experts_per_ep_rank * (p_state.ep_rank + 1)
                        else:
                            start, end = 0, model_tensor.shape[0]
                        v = v[start:end].contiguous()
                state_dict[new_k] = v

            # Assign tied weights if applicable
            if self._tie_word_embeddings:
                input_embeddings_key = self.get_input_embeddings_key()
                output_embeddings_key = self.get_output_embeddings_key()
                if isinstance(state_dict.get(input_embeddings_key, None), torch.Tensor) and not isinstance(
                        state_dict.get(output_embeddings_key, None), torch.Tensor):
                    state_dict[output_embeddings_key] = state_dict[input_embeddings_key].clone()
                    logger.debug(f"[rank {dist.get_rank()}] "
                                 f"Tied weight assigned: {output_embeddings_key} from {input_embeddings_key}",
                                 flush=True)

        state_dict = {k: load_fn for k in checkpoint_keys}
        return LazyStateDict(state_dict)

    @staticmethod
    def _check_is_directory(weight_path):
        weight_path = Path(weight_path)

        if not weight_path.exists():
            raise FileNotFoundError(f'Weight path {weight_path} does not exist.')

        if weight_path.is_file():
            raise NotADirectoryError(f'Weight path {weight_path} is not a directory.')

        return weight_path

    def get_key_mapping(self):
        # load_name1:model_name1, load_name2:model_name2, load3:model3|load4:model4 ...
        key_mapping = self.args.key_mapping
        if key_mapping is None:
            key_mapping = []

        assert all(":" in km for km in key_mapping), \
            f"Invalid key mapping format: {key_mapping}. Expected format: pattern:replacement"

        parsed_mapping = {}
        for km in key_mapping:
            # It's well compatible with single mapping or multiple mappings separated by '|'
            groups = km.split('|')
            keys, values = zip(*(map(lambda x: x.strip(), g.split(':')) for g in groups))
            parsed_mapping[keys] = values
        return parsed_mapping

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """
        Load weights into the model from an iterable of (name, tensor) tuples.
        This is used for updating weights online, e.g., during inference stage of RL training.
        """
        model_state_dict = self.state_dict()
        for name, load_tensor in weights:
            if name not in model_state_dict:
                # skip if the weight is not found in the model
                continue
            model_tensor = model_state_dict[name]
            if model_tensor.shape != load_tensor.shape:
                raise ValueError(f"Shape mismatch for weight {name}: "
                                 f"model tensor shape {model_tensor.shape} vs. "
                                 f"loaded tensor shape {load_tensor.shape}")
            if isinstance(load_tensor, torch.Tensor):
                model_tensor.data.copy_(load_tensor.data)
            else:
                raise ValueError(f"Unsupported tensor type in load_weights for {name}: {type(load_tensor)}")

    @staticmethod
    def _calc_piece_count(hf_weight, shard_threshold):
        shard_id = 0
        current_size = 0
        current_shard = {}
        for name, tensor in hf_weight.items():
            tensor_bytes = tensor.numel() * tensor.element_size()

            if current_size > 0 and current_size + tensor_bytes > shard_threshold:
                # save current piece
                shard_id += 1
                current_shard = {}
                current_size = 0

            current_shard[name] = tensor
            current_size += tensor_bytes

        if current_shard:
            shard_id += 1

        return shard_id

    def state_dict_to_hf(self, state_dict, save_dir: str | Path, shard_size_gb=5):
        """
        Save the given state dict to HuggingFace format in the specified directory.
        TODO(jarvizhang): include tokenizer, config, code, saving.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        sorted_state_dict = dict()
        for key in natsorted(state_dict.keys()):
            sorted_state_dict[key] = state_dict[key]

        index = {
            "metadata": {},
            "weight_map": {}
        }

        shard_threshold = shard_size_gb * 1024 * 1024 * 1024
        total_size = 0
        current_size = 0
        shard_id = 1
        current_shard = {}

        total_shard_count = self._calc_piece_count(sorted_state_dict, shard_threshold)
        for name, tensor in sorted_state_dict.items():
            tensor_bytes = tensor.numel() * tensor.element_size()

            # Check is a new shard is needed.
            # (current shard is not empty and adding the new tensor exceeds the threshold)
            if current_size > 0 and current_size + tensor_bytes > shard_threshold:
                # save current piece
                file_name = f"model-{shard_id:05d}-of-{total_shard_count:05d}.safetensors"

                file_full_path = save_dir / file_name
                save_file(current_shard, file_full_path)
                print(f"[rank {rank}] Saved {file_name}", flush=True)
                for k in current_shard:
                    index["weight_map"][k] = file_name
                shard_id += 1
                current_shard = {}
                current_size = 0

            # convert DTensor to local tensor if needed
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()

            current_shard[name] = tensor
            current_size += tensor_bytes
            total_size += tensor_bytes

        if current_shard:
            file_name = f"model-{shard_id:05d}-of-{total_shard_count:05d}.safetensors"
            file_full_path = save_dir / file_name
            if rank == 0:
                save_file(current_shard, file_full_path)
                print(f"[rank {rank}] Saved {file_name}", flush=True)
            for k in current_shard:
                index["weight_map"][k] = file_name

        index["metadata"]["total_size"] = total_size

        index_path = save_dir / HF_WEIGHT_INDEX_NAME
        config_path = save_dir / HF_CONFIG_NAME
        if rank == 0:
            with index_path.open("w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False)
            print(f"[rank {rank}] Saved {index_path}", flush=True)

            if not config_path.exists():
                # Save an empty config file to indicate the checkpoint is in HuggingFace format.
                with config_path.open("w", encoding="utf-8") as f:
                    json.dump({}, f)


class LoadModelWrapper(ModelWrapper):
    """ A model wrapper to concat expert weights into single tensor for FusedMoE layers. """
    def get_state_dict(self):
        """ Keys and values must be aligned with checkpoint. """
        args = get_args()
        original_state_dict = {
            k: v
            for sd in map(get_model_state_dict, self.model)
            for k, v in sd.items()
        }
        assert isinstance(self.model, list) and len(self.model) == 1, \
            "LoadModelWrapper currently only supports single model instance."
        config = self.model[0].get_config()
        assert not config.mlp_bias, "LoadModelWrapper currently only supports MLP without bias."

        key_renaming_mapping, key_mapping = self.model[0].get_model_key_renaming_mapping(original_state_dict.keys())
        original_state_dict = {
            key_renaming_mapping[k]: v
            for k, v in original_state_dict.items()
        }
        if len(key_mapping) > 0 and torch.distributed.get_rank() == 0:
            print("============================== Key Renaming Mapping ==============================")
            key_str_capacity = max([len(k) for k in key_renaming_mapping]) + 4
            for k, v in key_renaming_mapping.items():
                print(f"{k:<{key_str_capacity}s} {v}")
            print("=" * 82)

        if config.moe_impl == "flashinfer" and (not args.fuse_experts_in_load or config.split_gate_and_up):
            # fuse 且 not split 时，恰好是 flashinfer 的格式，不需要处理， 否则需要下面逻辑处理一下
            # 本段代码的 fuse_experts_in_load 和 hf 处表达的含义不完全一致, 
            # fuse_experts_in_load 本身的含义是加载的 ckpt 非 fused, 但创建的模型为 fused, 这种情况下需要传入此参数
            # 但本段的逻辑为：
            #   fuse_experts_in_load=True:  表示模型创建时是 fused, 是 ep_moe, flashinfer 和 ep_moe 共享内存
            #   fuse_experts_in_load=False: 表示模型创建时是 unfused, 是 deepseek, flashinfer 和 deepseek 共享内存
            # config 下的 split_gate_and_up 表示模型创建时是不是split的

            # If fuse_experts_in_load is enabled and config.split_gate_and_up is False, the checkpoint stores fused
            # expert weights with shape [num_experts, 2 * hidden_size, hidden_size], which already matches the weights
            # of flashinfer. In this case, we can directly load the checkpoint without rearranging the weights.

            # Rearrange expert weights for FusedMoE layers
            ignored_experts_keys = ["gate_and_up_proj", "down_proj", "gate_proj", "up_proj"]

            load_state_dict = {}
            for name, param in original_state_dict.items():
                if any("experts" in name and ik in name for ik in ignored_experts_keys):
                    assert param.numel() == 0, \
                        f"Expected empty tensor for ignored key {name}, but got shape {param.shape}."
                    continue

                if "expert_gate_and_up_weights" in name or "expert_down_weights" in name:
                    # FusedMoE expert weights are stored as a single tensor with shape [num_experts, ...]
                    prefix, suffix = name.rsplit(".", 1)

                    if args.fuse_experts_in_load: # 回转读取 EP MOE 的 dcp checkpoint 到 flashinfer 内存
                        if suffix == "expert_gate_and_up_weights":
                            ckpt_gate_key = f"{prefix}.experts.gate_proj_weights"
                            ckpt_up_key = f"{prefix}.experts.up_proj_weights"
                            load_state_dict[ckpt_gate_key] = param[:, config.moe_ffn_hidden_size:]
                            load_state_dict[ckpt_up_key] = param[:, :config.moe_ffn_hidden_size]
                        else:
                            ckpt_down_key = f"{prefix}.experts.down_proj_weights"
                            load_state_dict[ckpt_down_key] = param
                    else: # 回转读取 deepseek 的 dcp checkpoint 到 flashinfer 内存
                        for i in range(config.num_experts):
                            if suffix == "expert_gate_and_up_weights":
                                if config.split_gate_and_up:
                                    ckpt_gate_key = f"{prefix}.experts.{i}.gate_proj.weight"
                                    ckpt_up_key = f"{prefix}.experts.{i}.up_proj.weight"
                                    load_state_dict[ckpt_gate_key] = param[i:i + 1, config.moe_ffn_hidden_size:].view(
                                        config.moe_ffn_hidden_size, -1)
                                    load_state_dict[ckpt_up_key] = param[i:i + 1, :config.moe_ffn_hidden_size].view(
                                        config.moe_ffn_hidden_size, -1)
                                else:
                                    ckpt_gate_and_up_key = f"{prefix}.experts.{i}.gate_and_up_proj.weight"
                                    load_state_dict[ckpt_gate_and_up_key] = param[i:i + 1].view(*param.shape[1:])
                            else:
                                ckpt_down_key = f"{prefix}.experts.{i}.down_proj.weight"
                                load_state_dict[ckpt_down_key] = param[i:i + 1].view(*param.shape[1:])

                else:
                    load_state_dict[name] = param
        elif config.moe_impl == "ep_moe":
            if config.split_gate_and_up:
                raise NotImplementedError("ep_moe never splits gate and up")
            if args.fuse_experts_in_load:
                loguru.logger.warning("fuse_experts_in_load will be ignored for ep_moe, please ensure the checkpoint loaded is fused already.")
            load_state_dict = original_state_dict
        else:
            load_state_dict = original_state_dict

        return load_state_dict

    def load_state_dict(self, state_dict):
        # 1. dcp.load will load checkpoint into the state_dict inplace,
        # therefore there is no need to call model's load_state_dict again.
        pass


@dataclass
class HunyuanMultimodalOutput(ModelOutput):
    """
    Base class for hunyuan multimodal model (generalized autoregressive) outputs.

    Args:
        losses (dict[str, torch.Tensor], optional): A dictionary of loss components.
            Language modeling loss and diffusion loss are usually included.
        logits (torch.Tensor, optional): The prediction scores of the language modeling head.
        past_key_values (Cache, optional): Contains pre-computed hidden-states (key and values
            in the self-attention blocks) that can be used to speed up sequential decoding.
        diffusion_prediction (torch.Tensor, optional): The predicted noise or denoised images
            from the diffusion modeling.
    """

    losses: Optional[dict[str, torch.Tensor]] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    diffusion_prediction: Optional[torch.Tensor] = None


@dataclass
class ParameterCount:
    total: int
    trainable: int
    frozen: int


@dataclass
class ParameterSummary:
    num_of_tensors: int
    num_of_params: int
    param_counts: dict[torch.dtype, ParameterCount]
    children_param_counts: Optional[dict[str, "ParameterSummary"]] = None

    def as_table(self, header=True):
        lines = [
            f"======================= Parameter Summary =======================" if header else "",
            f"Total number of tensors: {self.num_of_tensors:,}",
            f"Total number of parameters: {self.num_of_params:,}",
            f"{'Dtype':<15} {'Total':<15} {'Trainable':<15} {'Frozen':<15}"
        ]
        for dtype, counts in self.param_counts.items():
            lines.append(f"{str(dtype):<15} {counts.total:<15,} {counts.trainable:<15,} {counts.frozen:<15,}")
        if self.children_param_counts:
            lines.append("=================== Children Parameter Summary ==================")
            for idx, (child_name, child_summary) in enumerate(self.children_param_counts.items()):
                if idx > 0:
                    lines.append("-" * 65)
                lines.append(f"Child Module: {child_name}")
                child_table = child_summary.as_table(header=False)
                child_lines = child_table.splitlines()
                indented_child_lines = ["  " + line for line in child_lines]
                lines.extend(indented_child_lines)
        lines.append("=" * 65 if header else "")
        return "\n".join(lines)

    @classmethod
    def from_module(cls, model: nn.Module, with_children: bool = True) -> "ParameterSummary":
        num_of_tensors = 0
        num_of_params = 0
        param_counts = {}
        children_param_counts = {}

        for param in model.parameters():
            single_param_count = param.numel()
            num_of_tensors += 1
            num_of_params += single_param_count
            dtype = param.dtype
            if dtype not in param_counts:
                param_counts[dtype] = ParameterCount(total=0, trainable=0, frozen=0)
            param_counts[dtype].total += single_param_count
            if param.requires_grad:
                param_counts[dtype].trainable += single_param_count
            else:
                param_counts[dtype].frozen += single_param_count

        if with_children:
            for name, child in model.named_children():
                children_param_counts[name] = ParameterSummary.from_module(child, with_children=False)

        return cls(
            num_of_tensors=num_of_tensors,
            num_of_params=num_of_params,
            param_counts=param_counts,
            children_param_counts=children_param_counts or None,
        )


def summary_params(module):
    print(ParameterSummary.from_module(module).as_table())
