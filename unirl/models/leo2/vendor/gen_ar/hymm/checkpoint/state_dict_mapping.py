import re
import torch
import os

from hymm.models.multimodal.hunyuan_multimodal_state import HunyuanMultimodalState
from hy_parallelism.checkpoint.state_dict_mapping import ParamEntryMappingFn, StateDictMappingFn

def get_key_mapping_recipe_args(sd_source, model_name, moe_impl, vae_condition):
    args = []

    def add_args(new_args):
        if isinstance(new_args, list):
            for k in new_args:
                add_args(k)
        else:
            if new_args in args:
                raise ValueError(f"Duplicate argument: {new_args}")
            args.append(new_args)


    if sd_source == "torch_dcp":
        return args
    elif sd_source == "ptm1-hf":
        if model_name == 'MoE-A13B':
            # key-mapping:
            #   - embed_tokens:__
            #   - language_model\.transformer\.ln_f:model.norm
            #   - language_model\.transformer\.wte:model.embed_tokens
            #   - 'language_model\.:'
            #   - vision_aligner_so:vit_aligner
            #   - 'vision_model_so:vit|layernorm_mlp\.:'
            add_args(['--key-mapping', 'moe_a13b_ptm1_hf_to_fsdp2']) # 不开 split-qkv 和 split-gate-and-up
        else:
            raise ValueError(f"Unexpected model name for key mapping recipe: {model_name}")
    elif sd_source == "std-hf":
        if model_name == 'MoE-A13B':
            # key-mapping:
            #   - vision_model:vit
            #   - vision_aligner:vit_aligner
            #   - ln_f:norm
            #   - wte:embed_tokens
            add_args(['--key-mapping', 'hyimage3_hf_to_fsdp2'])
        else:
            raise ValueError(f"Unexpected model name for key mapping recipe: {model_name}")
    elif sd_source == "ptm2-dcp":
        add_args(['--swap-gate-and-up', '--load-remove-prefix'])

        if model_name == 'MoE-A13B':
            # key-mapping:
            #   - 'model\.layers:model.transformer.layers
            #   |model>self_attn:self_attention
            #   |model>qkv_proj:linear_qkv
            #   |model>o_proj:linear_proj
            #   |model>input_layernorm\.weight:self_attention.linear_qkv.layer_norm_weight
            #   |model>post_attention_layernorm:pre_mlp_layernorm
            #   |model>query_layernorm:q_layernorm
            #   |model>key_layernorm:k_layernorm
            #   |model>gate\.wg:router
            #   |model>expert_gate_and_up_weights:experts.experts.linear_fc1.weight
            #   |model>expert_down_weights:experts.experts.linear_fc2.weight
            #   |model>shared_mlp\.gate_and_up_proj:shared_experts.linear_fc1
            #   |model>shared_mlp\.down_proj:shared_experts.linear_fc2'
            add_args(['--key-mapping', 'moe_a13b_ptm2_to_fsdp2'])
            assert moe_impl == 'flashinfer', f"这个 key-mapping 似乎只支持 flashinfer"
        elif model_name == 'MoE-A3B':
            # key-mapping:
            #   - 'model\.layers:model.transformer.layers
            #   |model>self_attn:self_attention
            #   |model>qkv_proj:linear_qkv
            #   |model>o_proj:linear_proj
            #   |model>input_layernorm\.weight:self_attention.linear_qkv.layer_norm_weight
            #   |model>post_attention_layernorm:pre_mlp_layernorm
            #   |model>query_layernorm:q_layernorm
            #   |model>key_layernorm:k_layernorm
            #   |model>gate\.wg:router
            #   |model>expert_gate_and_up_weights:experts.experts.linear_fc1.weight
            #   |model>expert_down_weights:experts.experts.linear_fc2.weight
            #   |model>shared_mlp\.gate_and_up_proj:shared_experts.linear_fc1
            #   |model>shared_mlp\.down_proj:shared_experts.linear_fc2
            #   |model>\.0\.pre_mlp_layernorm\.weight:.0.mlp.linear_fc1.layer_norm_weight
            #   |model>mlp\.gate_and_up_proj:mlp.linear_fc1
            #   |model>mlp\.down_proj:mlp.linear_fc2'

            add_args(['--key-mapping', 'moe_a3b_ptm2_to_fsdp2'])
            assert moe_impl == 'flashinfer', f"这个 key-mapping 似乎只支持 flashinfer"

            # qkv fused, gate and up fused
            # split-gate-and-up 有时候代表原始的参数状态，例如 dcp -> flash_infer 时，这代表这个 dcp 模型是否 split-gate-and-up 的
            # 涉及流程：pre_load_dcp, new_fsdp_impl 下的 flash transform
            add_args(['--no-split-qkv', '--no-split-gate-and-up'])

            if vae_condition:
                add_args([
                    '--cond-image-type', 'vit', 
                    '--cond-token-attn-type', 'causal', 
                ])
        else:
            raise ValueError(f"Unexpected model name for key mapping recipe: {model_name}")

        if moe_impl:
            add_args(['--moe-impl', moe_impl])

        if moe_impl == 'ep_moe':
            add_args(['--fuse-experts-in-load'])

    else:
        raise ValueError(f"Unexpected SD source for key mapping recipe: {sd_source}")
    return args


def maybe_extend_sd_vocab(state_dict, model_state_dict):
    class MaybeExtendVocabSD(ParamEntryMappingFn):

        def __init__(self, source_regex, model_state_dict):
            self.source_regex = source_regex
            self.model_state_dict = model_state_dict

        def mapped_keys(self):
            return [self.match.string]

        def value_fn(self, new_key, orig_key, match):
            sd = self.sd
            assert new_key == orig_key
            load_tensor = sd[orig_key]
            model_tensor = self.model_state_dict[orig_key]
            name = new_key

            if load_tensor.shape == model_tensor.shape:
                return load_tensor
            
            assert len(load_tensor.shape) == 2 and len(model_tensor.shape) == 2, \
                f"Expected 2D tensors for {name}, got {load_tensor.shape} and {model_tensor.shape}"
            assert load_tensor.shape[1] == model_tensor.shape[1], \
                f"{name} dimension mismatch for {name}: expected {model_tensor.shape[1]}, got {load_tensor.shape[1]}"
            assert load_tensor.shape[0] <= model_tensor.shape[0], \
                f"Cannot align {name}: loaded tensor size {load_tensor.shape[0]} " \
                f"is larger than model tensor size {model_tensor.shape[0]}"
            # Clone the model tensor as new tensor and copy the loaded values
            new_tensor = model_tensor.data.clone()
            new_tensor[:load_tensor.shape[0], :] = load_tensor
            return new_tensor

    return StateDictMappingFn(
        state_dict, [], 
        [MaybeExtendVocabSD("(model.embed_tokens.weight)|(lm_head.weight)", model_state_dict)], 
        exists_handler='skip'
    )()

def maybe_convert_deepseek_moe_to_flashinfer(state_dict, args, config):
    # deepseek/ep_moe experts -> flashinfer fused experts
    class DeepseekToFlashInferMoE(ParamEntryMappingFn):
        def __init__(self, source_regex):
            self.source_regex = source_regex

        def mapped_keys(self):
            prefix = self.match.group("prefix")
            if 'up_proj' in self.match.string or 'gate_proj' in self.match.string:
                return [f'{prefix}.expert_gate_and_up_weights']
            elif 'down_proj' in self.match.string:
                return [f'{prefix}.expert_down_weights']
            elif 'gate_and_up_proj' in self.match.string:
                return [f'{prefix}.expert_gate_and_up_weights']
            else:
                raise ValueError(f"Unexpected mapped key: {self.match.string}")

        def value_fn(self, new_key, orig_key, match):
            if os.environ.get('RANK') == '0':
                print(f'FlashInfer conversion:  {orig_key[-64:]:<64} -> {new_key[-64:]:<64}')

            prefix = match.group("prefix")
            suffix = new_key.rsplit(".", 1)[-1]

            sd = self.sd
            num_experts = config.num_experts

            if suffix == "expert_gate_and_up_weights":
                gate_and_up_weights = []
                for expert_idx in range(num_experts):
                    if config.split_gate_and_up:
                        up_key = f"{prefix}.experts.{expert_idx}.up_proj.weight"
                        gate_key = f"{prefix}.experts.{expert_idx}.gate_proj.weight"
                        assert up_key in sd and gate_key in sd, (
                            f"Missing MoE expert weights for {prefix}, expert_idx={expert_idx}: "
                            f"{up_key=} {gate_key=}"
                        )
                        gate_and_up_weights.append(torch.cat((sd[up_key], sd[gate_key]), dim=0).data)
                    else:
                        gate_and_up_key = f"{prefix}.experts.{expert_idx}.gate_and_up_proj.weight"
                        assert gate_and_up_key in sd, (
                            f"Missing MoE expert weight for {prefix}, expert_idx={expert_idx}: "
                            f"{gate_and_up_key=}"
                        )
                        gate_and_up_weights.append(sd[gate_and_up_key].data)

                # TODO: 这里可以顺便 swap，但为了统一，这里先不做
                return torch.stack(gate_and_up_weights).contiguous()

            if suffix == "expert_down_weights":
                down_weights = []
                for expert_idx in range(num_experts):
                    down_key = f"{prefix}.experts.{expert_idx}.down_proj.weight"
                    assert down_key in sd, (
                        f"Missing MoE expert down weight for {prefix}, expert_idx={expert_idx}: "
                        f"{down_key=}"
                    )
                    down_weights.append(sd[down_key].data)
                return torch.stack(down_weights).contiguous()

            raise ValueError(f"Unexpected mapped key: {new_key} from {orig_key}")

    return StateDictMappingFn(
        state_dict,
        [],
        [
            DeepseekToFlashInferMoE(
                source_regex=re.compile(
                    r"^(?P<prefix>.+)\.experts\.(?P<expert_idx>\d+)\.(?P<proj>up_proj|gate_proj|down_proj|gate_and_up_proj)\.weight$"
                )
            )
        ],
        exists_handler='skip'
    )()



def maybe_convert_flashinfer_to_hunyuan_experts(state_dict, args, config):
    # deepseek 或者 ep_moe (fused / unfused) -> flash_infer

    class FlashInferToDeepseekOrEpMoE(ParamEntryMappingFn):
        def __init__(self, source_regex):
            self.source_regex = source_regex

        def mapped_keys(self):
            prefix = self.match.group("prefix")
            suffix = self.match.group("suffix")

            fuse_experts_in_load = args.fuse_experts_in_load

            num_experts = config.num_experts
            if suffix == "expert_gate_and_up_weights":
                if fuse_experts_in_load:
                    return [
                        f"{prefix}.experts.gate_proj_weights",
                        f"{prefix}.experts.up_proj_weights",
                    ]
                if config.split_gate_and_up:
                    keys = []
                    for i in range(num_experts):
                        keys.extend(
                            [
                                f"{prefix}.experts.{i}.gate_proj.weight",
                                f"{prefix}.experts.{i}.up_proj.weight",
                            ]
                        )
                    return keys
                return [f"{prefix}.experts.{i}.gate_and_up_proj.weight" for i in range(num_experts)]

            if suffix == "expert_down_weights":
                if fuse_experts_in_load:
                    return [f"{prefix}.experts.down_proj_weights"]
                return [f"{prefix}.experts.{i}.down_proj.weight" for i in range(num_experts)]

            raise ValueError(f"Unexpected suffix for MoE mapping: {suffix}")

        def value_fn(self, new_key, orig_key, match):
            value = self.sd[orig_key]
            suffix = match.group("suffix")
            fuse_experts_in_load = args.fuse_experts_in_load
            split_gate_and_up = config.split_gate_and_up

            if suffix == "expert_gate_and_up_weights":
                hidden_size = config.moe_ffn_hidden_size
                if fuse_experts_in_load:
                    if new_key.endswith(".experts.gate_proj_weights"):
                        return value[:, hidden_size:]
                    if new_key.endswith(".experts.up_proj_weights"):
                        return value[:, :hidden_size]
                    raise ValueError(f"Unexpected mapped key: {new_key}")

                expert_idx = int(re.search(r"\.experts\.(\d+)\.", new_key).group(1))
                if split_gate_and_up:
                    if new_key.endswith(".gate_proj.weight"):
                        return value[expert_idx:expert_idx + 1, hidden_size:].reshape(hidden_size, -1)
                    if new_key.endswith(".up_proj.weight"):
                        return value[expert_idx:expert_idx + 1, :hidden_size].reshape(hidden_size, -1)
                    raise ValueError(f"Unexpected mapped key: {new_key}")

                return value[expert_idx:expert_idx + 1].reshape(*value.shape[1:])

            if suffix == "expert_down_weights":
                if fuse_experts_in_load:
                    return value
                expert_idx = int(re.search(r"\.experts\.(\d+)\.", new_key).group(1))
                return value[expert_idx:expert_idx + 1].reshape(*value.shape[1:])

            raise ValueError(f"Unexpected suffix for MoE mapping: {suffix}")

    return StateDictMappingFn(
        state_dict,
        [],
        [
            FlashInferToDeepseekOrEpMoE(
                source_regex=re.compile(
                    r"^(?P<prefix>.+)\.(?P<suffix>expert_gate_and_up_weights|expert_down_weights)$"
                ),
            )
        ],
        exists_handler='skip'
    )()

