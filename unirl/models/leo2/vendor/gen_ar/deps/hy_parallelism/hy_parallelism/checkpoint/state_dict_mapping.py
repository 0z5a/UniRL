r"""
使用方法:
1. 创建规则:
    mapping_fn = StateDictMappingFn(
        state_dict, 
        [
            (r'^(.+)\.gate\.wg\.weight$', r'\1.router.gate.weight'), # 简单正则映射规则
        ], 
        [
            # 复杂规则，继承 ParamEntryMappingFn 实现
            # 遍历 state_dict, 匹配上正则的，会调用 mapped_keys, 获取 key 列表 (因为 key 可能被拆分)
            # 然后调用 value_fn, 获取新 param
            # 在 mapped_keys 和 value_fn 中，可以访问 self.sd, self.original_key, self.match
            MaybeExtendVocabSD("(model.embed_tokens.weight)|(lm_head.weight)", model)
        ], 
        exists_handler='skip'
    )
2. 映射, 获取 lazy state_dict
    new_state_dict = mapping_fn(state_dict)

"""



import loguru
import re
from functools import partial
from typing import Dict, Any, Type
import torch
import copy
from itertools import count

from .lazy_state_dict import LazyMappedStateDict, LazyEntry

class ParamEntryMappingFn(object):
    """
    初始化设定一个 regex
    给定一个 key, setup_params 进行匹配
    mapped_keys 将这个key转换或拆分成多个新 key (根据setup的东西, 比如self.match)
    value_fn 针对每一个新key, 实现转换 tensor 的逻辑
    call 获得一个 key -> callable 的 dict
    """

    def __init__(self, source_regex):
        self.source_regex = source_regex

    def mapped_keys(self) -> list[str]:
        """
        实现这个regex负责生产多少个新key
        一个 old key 可能被拆分成多个新key, 这里需要全部返回
        会根据 setup_params 种的 match 来判断
        """
        raise NotImplementedError
        # return [self.match.string]

    def value_fn(self, new_key, orig_key, match: re.Match):
        """
        实现这个 regex 拿到新旧key的转换时, 应该怎么转换tensor
        """
        raise NotImplementedError

    def regex_match(self, k: str) -> re.Match | None:
        if isinstance(self.source_regex, re.Pattern):
            return self.source_regex.match(k)
        elif isinstance(self.source_regex, str):
            return re.match(self.source_regex, k)
        else:
            raise ValueError(f'Invalid source_regex type: {type(self.source_regex)}')
    
    def setup_params(self, sd: dict, original_key: str, match: re.Match):
        self.sd = sd
        self.original_key = original_key
        self.match = match

    def __call__(self) -> Any:
        keys = self.mapped_keys()
        ret = {}
        for key in keys:
            ret[key] = LazyEntry(key, partial(self.value_fn, new_key=key, orig_key=self.original_key, match=self.match))
        return ret

class StateDictMappingFn(object):

    def __init__(self, sd: dict, regex_expand_pairs, regex_fn_pairs, exists_handler='skip'):
        self.sd = sd
        self.regex_expand_pairs = regex_expand_pairs
        self.regex_fn_pairs = regex_fn_pairs
        self.exists_handler = exists_handler
        self.check_is_valid_source = lambda: True

    def __call__(self) -> LazyMappedStateDict:
        assert self.check_is_valid_source(), f'Invalid source state_dict: {self.sd}'
        return self.param_mapping_fn_template(self.sd, self.regex_expand_pairs, self.regex_fn_pairs, self.exists_handler)

    @staticmethod
    def param_mapping_fn_template(sd: dict, regex_expand_pairs, regex_fn_pairs, exists_handler='skip') -> LazyMappedStateDict:
        # exists_handler: 'check' | 'skip'
        ret = {}
        
        def get_value(orig_key):
            return sd[orig_key]

        def merge_dicts_to_ret(new_dict: dict):
            for k, v in new_dict.items():
                if k in ret:
                    if exists_handler == 'check':
                        if callable(ret[k]) or callable(v):
                            raise NotImplementedError(f'callable value for {exists_handler=} is not supported')
                        torch.testing.assert_close(ret[k], v)
                    elif exists_handler == 'skip':
                        continue
                ret[k] = v
            return True


        copy_keys = copy.deepcopy(list(sd.keys()))
        for k in copy_keys:
            if k not in sd: # if key is not in sd, it means it has been processed
                raise ValueError(f'Can not remove entry from state_dict during iteration.')
                continue
            for entry_transformer in regex_fn_pairs:
                if isinstance(entry_transformer, ParamEntryMappingFn):
                    if match := entry_transformer.regex_match(k):
                        entry_transformer.setup_params(sd, k, match)
                        new_dict = entry_transformer()
                        merge_dicts_to_ret(new_dict)
                        break
                else: # Old implementation
                    pattern, fn = entry_transformer
                    if match := re.match(pattern, k):
                        new_dict = fn(sd, k, match)
                        merge_dicts_to_ret(new_dict)
                        break
            else:
                # Handle other MoE patterns
                for pattern, replacement in regex_expand_pairs:
                    if match := re.match(pattern, k):
                        ret[match.expand(replacement)] = partial(get_value, k)
                        break
                else:
                    # Non-MoE keys: copy directly
                    ret[k] = None
        
        return LazyMappedStateDict(sd, ret)


def convert_hunyuan_a3b_vlm_to_hunyuanimage_35(sd: dict) -> LazyMappedStateDict:
    # load: /apdcephfs_zwfy/share_303937731/1_public_models/hymm_ar_assets/pretrained_llm/raw_vlm/hunyuan_a3b_vlm_pretrain_stage2_500B/hf
    raise NotImplementedError

def convert_hyimage35_puretorch_to_titan_moe(sd: dict) -> LazyMappedStateDict:
    """ ar dev; puretorch checkpoint to titan moe """

    class ExpertsParamEntryMappingFn(ParamEntryMappingFn):
        reg = r'\1.experts.<expert_idx>.\3.weight'
        weight_type_group = 3

        def mapped_keys(self) -> list[str]:
            match = self.match

            prefix = match.group(1)
            weight_type = match.group(self.weight_type_group)
            if weight_type == 'gate_proj':
                return [f"{prefix}.experts.w1"]
            elif weight_type == 'up_proj':
                return [f"{prefix}.experts.w3"]
            elif weight_type == 'down_proj':
                return [f"{prefix}.experts.w2"]
            elif weight_type == 'gate_and_up_proj':
                return [f"{prefix}.experts.w1", f"{prefix}.experts.w3"]
            else:
                raise ValueError(f"Unknown weight type: {weight_type}")

        def value_fn(self, new_key, orig_key, match: re.Match):
            reg = self.reg
            sd = self.sd
            weight_type = match.group(self.weight_type_group)

            weights = []
            for i in count():
                expert_i = match.expand(reg.replace('<expert_idx>', str(i)))
                # print(f'{match=}  {k=}: Try to get expert {expert_i}')
                if expert_i not in sd:
                    break
                weights.append(sd[expert_i])

            if weight_type == 'gate_proj':
                return torch.stack(weights, dim=0)
            elif weight_type == 'up_proj':
                return torch.stack(weights, dim=0)
            elif weight_type == 'down_proj':
                return torch.stack(weights, dim=0)
            elif weight_type == 'gate_and_up_proj':
                gate_and_up_weight = torch.stack(weights, dim=0)
                if 'w1' in new_key:
                    return gate_and_up_weight[:gate_and_up_weight.shape[1] // 2]
                elif 'w3' in new_key:
                    return gate_and_up_weight[gate_and_up_weight.shape[1] // 2:]
                else:
                    raise ValueError(f"Unknown weight type: {weight_type}")
            else:
                raise ValueError(f"Unknown weight type: {weight_type}")

    class SharedExpertsParamEntryMappingFn(ParamEntryMappingFn):
        # source_regex: r'^(.+)\.shared_mlp\.(.+)\.weight$'
        reg = r'\1.shared_experts.\2.weight'
        weight_type_group = 2

        def mapped_keys(self) -> list[str]:
            match = self.match
            prefix = match.group(1)

            weight_type = match.group(self.weight_type_group)
            if weight_type == 'gate_proj':
                return [f"{prefix}.shared_experts.w1.weight"]
            elif weight_type == 'up_proj':
                return [f"{prefix}.shared_experts.w3.weight"]
            elif weight_type == 'down_proj':
                return [f"{prefix}.shared_experts.w2.weight"]
            elif weight_type == 'gate_and_up_proj':
                return [f"{prefix}.shared_experts.w1.weight", f"{prefix}.shared_experts.w3.weight"]
            else:
                raise ValueError(f"Unknown weight type: {weight_type}")

        def value_fn(self, new_key, orig_key, match: re.Match):
            sd = self.sd
            weight_type = match.group(self.weight_type_group)

            weights = sd[orig_key]

            if weight_type == 'gate_proj':
                return weights
            elif weight_type == 'up_proj':
                return weights
            elif weight_type == 'down_proj':
                return weights
            else:
                raise ValueError(f"Unknown weight type: {weight_type}")
    
    return StateDictMappingFn(
        sd, 
        [
            (r'^(.+)\.gate\.wg\.weight$', r'\1.router.gate.weight'),
            # (r'vit\.vit\.(.+)', r'vit.encoder.\1'), 
        ], 
        [
            ExpertsParamEntryMappingFn(source_regex=re.compile(r'^(.+)\.experts\.(\d+)\.(.+)\.weight$')),
            SharedExpertsParamEntryMappingFn(source_regex=re.compile(r'^(.+)\.shared_mlp\.(.+)\.weight$')),
        ], 
        exists_handler='skip'
    )()




def convert_torch_to_titan_moe_mapping_fn_v2(sd: dict) -> LazyMappedStateDict:

    def gate_up_pattern_fn(sd: dict, k: str, match: re.Match) -> Dict[str, Any]:
        ret = {}
        prefix = match.group(1)
        def get_w1(orig_key):
            weight = sd[orig_key]
            return weight[weight.shape[0] // 2:]
        
        def get_w3(orig_key):
            weight = sd[orig_key]
            return weight[:weight.shape[0] // 2]
        ret[f"{prefix}.shared_experts.w1.weight"] = partial(get_w1, k)
        ret[f"{prefix}.shared_experts.w3.weight"] = partial(get_w3, k)
        return ret

    moe_patterns = [
        (r'^(.+)\.gate\.wg\.weight$', r'\1.router.gate.weight'),
        (r'^(.+)\.experts\.gate_proj$', r'\1.experts.w1'),
        (r'^(.+)\.experts\.up_proj$', r'\1.experts.w3'),
        (r'^(.+)\.experts\.down_proj$', r'\1.experts.w2'),
        (r'^(.+)\.shared_mlp\.down_proj\.weight$', r'\1.shared_experts.w2.weight'),
    ]
    gate_up_pattern = re.compile(r'^(.+)\.shared_mlp\.gate_and_up_proj\.weight$')
    return StateDictMappingFn(
        sd,
        moe_patterns,
        [ (gate_up_pattern, gate_up_pattern_fn), ]
    )()
    


def convert_torch_to_titan_moe_mapping_fn(sd: dict) -> LazyMappedStateDict:
    """
    Torch MOE v1 (Veomini) MOE to titan moe
    
    Returns a dictionary mapping new keys to either None (direct copy) or callables.
    Callables are called without arguments, all parameters are bound via partial.
    
    Conversion rules:
    - {prefix}.gate.wg.weight -> {prefix}.router.gate.weight
    - {prefix}.experts.gate_proj -> {prefix}.experts.w1
    - {prefix}.experts.up_proj -> {prefix}.experts.w3
    - {prefix}.experts.down_proj -> {prefix}.experts.w2
    - {prefix}.shared_mlp.gate_and_up_proj.weight -> split into w1 and w3
    - {prefix}.shared_mlp.down_proj.weight -> {prefix}.shared_experts.w2.weight
    
    Non-MoE keys are preserved unchanged.
    """
    ret = {}
    
    moe_patterns = [
        (r'^(.+)\.gate\.wg\.weight$', r'\1.router.gate.weight'),
        (r'^(.+)\.experts\.gate_proj$', r'\1.experts.w1'),
        (r'^(.+)\.experts\.up_proj$', r'\1.experts.w3'),
        (r'^(.+)\.experts\.down_proj$', r'\1.experts.w2'),
        (r'^(.+)\.shared_mlp\.down_proj\.weight$', r'\1.shared_experts.w2.weight'),
    ]
    gate_up_pattern = re.compile(r'^(.+)\.shared_mlp\.gate_and_up_proj\.weight$')
    
    def get_value(orig_key):
        return sd[orig_key]
    
    def get_w1(orig_key):
        weight = sd[orig_key]
        return weight[weight.shape[0] // 2:]
    
    def get_w3(orig_key):
        weight = sd[orig_key]
        return weight[:weight.shape[0] // 2]
    
    for k in sd.keys():
        # if match := gate_up_pattern.match(k):
        if match := re.match(gate_up_pattern, k):
            prefix = match.group(1)
            ret[f"{prefix}.shared_experts.w1.weight"] = partial(get_w1, k)
            ret[f"{prefix}.shared_experts.w3.weight"] = partial(get_w3, k)
        else:
            # Handle other MoE patterns
            for pattern, replacement in moe_patterns:
                if match := re.match(pattern, k):
                    ret[match.expand(replacement)] = partial(get_value, k)
                    break
            else:
                # Non-MoE keys: copy directly
                ret[k] = None
    
    return LazyMappedStateDict(sd, ret)

