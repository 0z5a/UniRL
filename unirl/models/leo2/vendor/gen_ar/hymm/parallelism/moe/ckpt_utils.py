# python3 -m hymm.parallelism.moe.ckpt_utils
import argparse
import json
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm


class GeminiToTorch:
    def __init__(self):
        self.experts_weights_w1 = []
        self.experts_weights_w3 = []
        self.experts_weights_w2 = []
        self.keys = []
        self.pattern = r'experts\.(\d+)\.'

    def gate_and_up_to_w(self, param):
        # gate_and_up: w3, w1
        # down: w2
        w1, w3 = param.chunk(2, dim=0)
        return w3, w1

    def transform(self, model_state_dict):
        self.experts_weights_w1 = []
        self.experts_weights_w2 = []
        self.experts_weights_w3 = []
        self.keys = [key for key in model_state_dict.keys()]

        for key in self.keys:
            if 'gate.wg.weight' in key:
                prefix = key[:key.find('gate.wg.weight')]
                new_key = key.replace('gate.wg.weight', 'router.gate.weight')
                w = model_state_dict[key]
                model_state_dict[new_key] = w
            elif 'shared_mlp.gate_and_up_proj.weight' in key:
                new_key1 = key.replace('shared_mlp.gate_and_up_proj.weight', 'shared_expert.w1')
                new_key2 = key.replace('shared_mlp.gate_and_up_proj.weight', 'shared_expert.w3')
                w3, w1 = self.gate_and_up_to_w(model_state_dict[key])
                model_state_dict[new_key1] = w1.unsqueeze(0)
                model_state_dict[new_key2] = w3.unsqueeze(0)
            elif 'shared_mlp.down_proj.weight' in key:
                new_key = key.replace('shared_mlp.down_proj.weight', 'shared_expert.w2')
                w = model_state_dict[key]
                model_state_dict[new_key] = w.unsqueeze(0)
            elif 'gate_and_up_proj.weight' in key:
                match = re.search(self.pattern, key)
                expert_id = int(match.group(1))
                w3, w1 = self.gate_and_up_to_w(model_state_dict[key])
                self.experts_weights_w1.append(w1)
                self.experts_weights_w3.append(w3)
            elif 'down_proj.weight' in key:
                match = re.search(self.pattern, key)
                expert_id = int(match.group(1))
                self.experts_weights_w2.append(model_state_dict[key])

            model_state_dict.pop(key)

        self.experts_weights_w1 = torch.stack(self.experts_weights_w1, dim=0)
        self.experts_weights_w3 = torch.stack(self.experts_weights_w3, dim=0)
        self.experts_weights_w2 = torch.stack(self.experts_weights_w2, dim=0)

        model_state_dict[prefix + 'experts.w1'] = self.experts_weights_w1
        model_state_dict[prefix + 'experts.w3'] = self.experts_weights_w3
        model_state_dict[prefix + 'experts.w2'] = self.experts_weights_w2
        return model_state_dict


class GeminiDefaultToGeminiEP:
    def __init__(self):
        self.keys = []
        self.gate_weights = {}
        self.up_weights = {}
        self.down_weights = {}
        self.pattern = r'experts\.(\d+)\.'

    def gate_and_up_split(self, param):
        up, gate = param.chunk(2, dim=0)
        return gate, up

    def transform(self, model_state_dict, has_bias=False):
        model_state_dict = model_state_dict.copy()

        self.keys = [key for key in model_state_dict.keys()]
        targets = ['weight']
        if has_bias:
            targets.append('bias')
        for target in targets:
            self.gate_weights = {}
            self.up_weights = {}
            self.down_weights = {}
            for key in self.keys:
                match = re.search(self.pattern, key)
                if match:
                    expert_id = int(match.group(1))
                    prefix = key[:key.find('experts.') + len('experts.')]
                    if f'gate_and_up_proj.{target}' in key:
                        gate, up = self.gate_and_up_split(model_state_dict[key])
                        if prefix not in self.gate_weights.keys():
                            self.gate_weights[prefix] = {}
                        if prefix not in self.up_weights.keys():
                            self.up_weights[prefix] = {}
                        self.gate_weights[prefix][expert_id] = gate
                        self.up_weights[prefix][expert_id] = up
                        model_state_dict.pop(key)
                    elif f'down_proj.{target}' in key:
                        if prefix not in self.down_weights.keys():
                            self.down_weights[prefix] = {}
                        self.down_weights[prefix][expert_id] = model_state_dict[key]
                        model_state_dict.pop(key)

            for prefix in self.gate_weights.keys():
                num_experts = len(self.gate_weights[prefix])
                wg = []
                wu = []
                wd = []
                for _ in range(num_experts):
                    wg.append(self.gate_weights[prefix][_])
                    wu.append(self.up_weights[prefix][_])
                    wd.append(self.down_weights[prefix][_])

                wg = torch.stack(wg, dim=0)
                wu = torch.stack(wu, dim=0)
                wd = torch.stack(wd, dim=0)
                if target == 'weight':
                    model_state_dict[prefix + 'gate_proj'] = wg
                    model_state_dict[prefix + 'up_proj'] = wu
                    model_state_dict[prefix + 'down_proj'] = wd
                else:
                    model_state_dict[prefix + 'gate_bias'] = wg
                    model_state_dict[prefix + 'up_bias'] = wu
                    model_state_dict[prefix + 'down_bias'] = wd

        return model_state_dict


def read_hf_file(ckpt_dir):
    weights = {}
    files = sorted(list(Path(ckpt_dir).glob('*.bin')))
    # 过滤 864 字节的空文件
    files = [f for f in files if f.stat().st_size > 864]
    n_bin_file = len(files)
    logger.info(f'reading {n_bin_file} files from {ckpt_dir}')
    for f in tqdm(files):
        weights.update(torch.load(f, map_location='cpu'))
    logger.info(f'{n_bin_file} files loaded')
    return weights


def hf_to_torch(
        weight,
        num_layers=32,
        n_expert=64
):
    # language_model.ptm_transformer.lm_head.weight / language_model.lm_head.weight             1
    # language_model.transformer.wte.weight / model.embed_tokens.weight                         0
    # language_model.transformer.ln_f.weight / model.norm.weight                                0

    new_weight = weight.copy()

    new_weight.pop("model.embed_tokens.weight")  # language_model.transformer.wte.weight
    if 'model.norm.weight' in new_weight:
        new_weight.pop("model.norm.weight") # language_model.transformer.ln_f.weight
    if 'model.norm.bias' in new_weight:
        new_weight.pop("model.norm.bias") # language_model.transformer.ln_f.bias
    new_weight.pop('language_model.ptm_transformer.lm_head.weight')

    for key in list(new_weight.keys()):
        if '.layernorm_mlp.' in key:
            new_weight[key.replace('.layernorm_mlp.', '.')] = new_weight.pop(key)

    for i in range(num_layers):
        new_weight[f"language_model.transformer.h.{i}.attn.attn.weight"] = new_weight.pop(f"model.layers.{i}.self_attn.qkv_proj.weight")
        new_weight[f"language_model.transformer.h.{i}.attn.proj.weight"] = new_weight.pop(f"model.layers.{i}.self_attn.o_proj.weight")
        new_weight[f"language_model.transformer.h.{i}.attn.q_norm.weight"] = new_weight.pop(f"model.layers.{i}.self_attn.query_layernorm.weight")
        new_weight[f"language_model.transformer.h.{i}.attn.k_norm.weight"] = new_weight.pop(f"model.layers.{i}.self_attn.key_layernorm.weight")
        new_weight[f"language_model.transformer.h.{i}.mlp.gate.wg.weight"] = new_weight.pop(f"model.layers.{i}.mlp.gate.wg.weight")
        new_weight[f"language_model.transformer.h.{i}.mlp.shared_mlp.gate_and_up_proj.weight"] = new_weight.pop(f"model.layers.{i}.mlp.shared_mlp.gate_and_up_proj.weight")
        new_weight[f"language_model.transformer.h.{i}.mlp.shared_mlp.down_proj.weight"] = new_weight.pop(f"model.layers.{i}.mlp.shared_mlp.down_proj.weight")
        for j in range(n_expert):
            new_weight[f"language_model.transformer.h.{i}.mlp.experts.{j}.gate_and_up_proj.weight"] = new_weight.pop(f"model.layers.{i}.mlp.experts.{j}.gate_and_up_proj.weight")
            new_weight[f"language_model.transformer.h.{i}.mlp.experts.{j}.down_proj.weight"] = new_weight.pop(f"model.layers.{i}.mlp.experts.{j}.down_proj.weight")

        new_weight[f"language_model.transformer.h.{i}.norm_1.weight"] = new_weight.pop(f"model.layers.{i}.input_layernorm.weight")
        new_weight[f"language_model.transformer.h.{i}.norm_2.weight"] = new_weight.pop(f"model.layers.{i}.post_attention_layernorm.weight")

    return new_weight


def torch_to_hf(
        weight,
        num_layers=32,
        n_expert=64
):
    new_weight = {}
    # vision_model_so.*                     # 448
    # vision_aligner_so.*                   # 4
    # language_model.timestep_emb.*         # 4
    # language_model.patch_embed.*          # 14
    # language_model.time_embed.*           # 4
    # language_model.time_embed_2.*         # 4
    # language_model.final_layer.*          # 16
    # language_model.lm_head.*              # 1
    # language_model.transformer.wte.*      # 1
    # language_model.transformer.ln_f.*     # 1
    # transformer.h.*                       # (9+64*2)*32
    # ------------------------------------------------
    # 448 + 4 + 4 + 14 + 4 + 4 + 16 + 1 + 1 + 1 + (9 + 64 * 2) * 32 = 4881

    for key in weight['model'].keys():
        if 'vision_model_so.encoder' in key and '.mlp.' in key:
            new_weight[key.replace('.mlp.', '.layernorm_mlp.mlp.')] = weight['model'][key]
        else:
            new_weight[key] = weight['model'][key]

    for i in range(num_layers):
        new_weight[f"model.layers.{i}.self_attn.qkv_proj.weight"] = new_weight.pop(f"language_model.transformer.h.{i}.attn.attn.weight")
        new_weight[f"model.layers.{i}.self_attn.o_proj.weight"] = new_weight.pop(f"language_model.transformer.h.{i}.attn.proj.weight")
        new_weight[f"model.layers.{i}.self_attn.query_layernorm.weight"] = new_weight.pop(f"language_model.transformer.h.{i}.attn.q_norm.weight")
        new_weight[f"model.layers.{i}.self_attn.key_layernorm.weight"] = new_weight.pop(f"language_model.transformer.h.{i}.attn.k_norm.weight")
        new_weight[f"model.layers.{i}.mlp.gate.wg.weight"] = new_weight.pop(f"language_model.transformer.h.{i}.mlp.gate.wg.weight")
        new_weight[f"model.layers.{i}.mlp.shared_mlp.gate_and_up_proj.weight"] = new_weight.pop(f"language_model.transformer.h.{i}.mlp.shared_mlp.gate_and_up_proj.weight")
        new_weight[f"model.layers.{i}.mlp.shared_mlp.down_proj.weight"] = new_weight.pop(f"language_model.transformer.h.{i}.mlp.shared_mlp.down_proj.weight")

        gate_proj = new_weight.pop(f"language_model.transformer.h.{i}.mlp.experts.gate_proj")   # [64, 3072, 4096]
        up_proj = new_weight.pop(f"language_model.transformer.h.{i}.mlp.experts.up_proj")       # [64, 3072, 4096]
        down_prob = new_weight.pop(f"language_model.transformer.h.{i}.mlp.experts.down_proj")   # [64, 4096, 3072]
        # Split by expert dim, then merge gate and up
        for j in range(n_expert):
            # HunyuanExpert impl:
            #   up * act_fn(gate)
            #
            # HunyuanMLP impl:
            #   x1, x2 = gate_and_up_prob.chunk(2)
            #   x1 * act_fn(x2)
            new_weight[f"model.layers.{i}.mlp.experts.{j}.gate_and_up_proj.weight"] = torch.cat([up_proj[j], gate_proj[j]], dim=0)  # [6144, 4096]
            new_weight[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = down_prob[j]     # [4096, 3072]

        new_weight[f"model.layers.{i}.input_layernorm.weight"] = new_weight.pop(f"language_model.transformer.h.{i}.norm_1.weight")
        new_weight[f"model.layers.{i}.post_attention_layernorm.weight"] = new_weight.pop(f"language_model.transformer.h.{i}.norm_2.weight")

    return new_weight


def save_sharded_hf(state_dict, save_dir, shard_size=4 * 1024 * 1024 * 1024):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    current_shard = OrderedDict()
    current_size = 0
    weight_map = {}
    pbar = tqdm(total=39, desc="Saving shards", unit="shard")

    for name, param in state_dict.items():
        param_size = param.numel() * param.element_size()

        if current_size + param_size > shard_size and current_shard:
            shard_name = f"pytorch_model-{len(shards) + 1:05d}-of-00039.bin"
            torch.save(current_shard, save_dir / shard_name)
            pbar.update(1)
            shards.append(shard_name)
            for name_, param_ in current_shard.items():
                weight_map[name_] = shard_name
            current_shard = OrderedDict()
            current_size = 0

        current_shard[name] = param
        current_size += param_size

    if current_shard:
        shard_name = f"pytorch_model-{len(shards) + 1:05d}-of-00039.bin"
        torch.save(current_shard, save_dir / shard_name)
        pbar.update(1)
        shards.append(shard_name)
        for name_, param_ in current_shard.items():
            weight_map[name_] = shard_name

    # 创建 index.json
    index = {
        "metadata": {"total_size": sum(p.numel() * p.element_size() for p in state_dict.values())},
        "weight_map": weight_map,
    }

    with (save_dir / "pytorch_model.bin.index.json").open("w") as f:
        json.dump(index, f, indent=4)


def gemini80b_transform_hf2dcp(hf_model_path, dcp_save_path):
    # source: /apdcephfs_nj10/share_301739632/2_public_experiments/log_EXP_gemini_beta/ckczzjzhang/ptm/7b_moe_pretrain_stage3_3/checkpoint/global_step15000/hf
    # hf_model_path = '/apdcephfs_zwfy/share_303937731/kevinkhwu/pretrain/hf'
    # dcp_save_path = '/apdcephfs_zwfy/share_303937731/kevinkhwu/pretrain/ep_dcp/weights'
    from hymm.parallelism.checkpoint_manager import torch_state_dict_to_dcp

    weights = read_hf_file(hf_model_path)
    weights_torch = hf_to_torch(weights)
    weights_torch_ep = GeminiDefaultToGeminiEP().transform(weights_torch)

    # assert len(weights) == len(weights_torch)
    # print('\n'.join(weights))
    # print('\n'.join(weights_torch))
    print(f"Total weights: {len(weights_torch_ep)}")
    logger.info(f"Saving DCP model to {dcp_save_path}")
    torch_state_dict_to_dcp(dcp_save_path, sd_input=weights_torch_ep)


def gemini80b_transform_dcp2hf(dcp_model_path, hf_save_path):
    from hymm.parallelism.checkpoint_manager import dcp_to_torch_state_dict

    logger.info(f"Loading DCP model from {dcp_model_path}")
    weights_torch = dcp_to_torch_state_dict(dcp_model_path)
    logger.info(f"Converting to HF format")
    weights_hf = torch_to_hf(weights_torch)
    # Add two useless weights for ptm compatibility
    weights_hf["model.embed_tokens.weight"] = weights_hf["language_model.transformer.wte.weight"][:128256, :]
    weights_hf["language_model.ptm_transformer.lm_head.weight"] = weights_hf["language_model.lm_head.weight"][:128256, :]
    logger.info(f"Total weights: {len(weights_hf)}")

    logger.info(f"Saving HF model to {hf_save_path}")
    Path(hf_save_path).mkdir(parents=True, exist_ok=True)
    save_sharded_hf(weights_hf, hf_save_path)

    logger.info(f"Saving config.json, configuration_hunyuan.py and hf_to_ptm_mapping.json")
    proj_root = Path(__file__).parents[2]
    config_json_file = proj_root / "models/autoregressive/hunyuan_moe_hf_config.json"
    config_define_file = proj_root / "models/autoregressive/configuration_hunyuan.py"
    hf_to_ptm_mapping = proj_root / "parallelism/moe/hf_to_ptm_mapping.json"
    shutil.copyfile(config_json_file, Path(hf_save_path) / "config.json")
    shutil.copyfile(config_define_file, Path(hf_save_path) / "configuration_hunyuan.py")
    shutil.copyfile(hf_to_ptm_mapping, Path(hf_save_path) / "hf_to_ptm_mapping.json")
    logger.info(f"Saved")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--convert-type", type=str, required=True,
                        choices=["dcp2hf", "hf2dcp"], help="Convert type.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input path.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output path.")
    args = parser.parse_args()
    return args


def test_case():
    # test value loss
    from hymm.parallelism.moe import gemini_moe
    from typing import Union, Literal, List
    from dataclasses import dataclass
    from accelerate.utils import set_seed

    @dataclass
    class Config():
        # MoE config
        n_expert: Union[int, List[int]] = 0
        n_expert_per_token: int = 0  # this is not used in HunYuanMoE
        hidden_act: Literal["silu", "gelu"] = "silu"
        use_mixed_mlp_moe: bool = False
        moe_intermediate_size: Union[int, List[int]] = None
        num_shared_expert: Union[int, List[int]] = None
        moe_topk: Union[int, List[int]] = None
        moe_drop_tokens: bool = False
        moe_random_routing_dropped_token: bool = False
        routed_scaling_factor: float = 1.0
        norm_topk_prob: bool = False
        n_group: bool = False
        topk_group: bool = False
        group_limited_greedy: bool = False
        n_embd: int = 1024
        intermediate_size: Union[int, List[int]] = None
        mlp_bias: bool = False,
        launcher: str = 'asdf'

    config = Config(
        moe_topk=4,
        n_embd=1024,
        intermediate_size=1024,
        moe_intermediate_size=1024,
        n_expert=16,
        mlp_bias=False,
        launcher='asdf',
    )

    gemini_ep = gemini_moe.HunYuanMoE(config, expert_plan='ep')

    gemini_default = gemini_moe.HunYuanMoE(config)

    model_trans2 = GeminiDefaultToGeminiEP()
    gemini_ep.load_state_dict(model_trans2.transform(gemini_default.state_dict()))

    # 模型随机浮点数输入测试

    hidden_act = torch.randn(10, 128, 1024)
    output1 = gemini_ep.forward(hidden_act)
    output2 = gemini_default.forward(hidden_act)
    rmse = torch.sqrt(torch.mean((output1 - output2) ** 2))
    assert torch.allclose(output1, output2, atol=1e-6)

    class WrapperModel1(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, expert_plan='default', gate_plan='default', shard_ep_in_ini=False):
            super(WrapperModel1, self).__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.model = gemini_moe.HunYuanMoE(config, expert_plan=expert_plan, gate_plan=gate_plan, shard_ep_in_ini=shard_ep_in_ini)
            self.modelx = gemini_moe.HunYuanMoE(config, expert_plan=expert_plan, gate_plan=gate_plan, shard_ep_in_ini=shard_ep_in_ini)
            self.linear2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.linear1(x)
            x = self.model(x)
            x = self.modelx(x)
            x = self.linear2(x)
            return x

    model1 = WrapperModel1(1024, 1024, 1024)
    model2 = WrapperModel1(1024, 1024, 1024, expert_plan='ep')

    model2.load_state_dict(GeminiDefaultToGeminiEP().transform(model1.state_dict()))

    set_seed(2)
    x = torch.randn(10, 128, 1024)
    output1 = model1(x)
    output2 = model2(x)
    print(output1)
    print(output2)
    rmse = torch.sqrt(torch.mean((output1 - output2) ** 2))
    assert torch.allclose(output1, output2, atol=1e-6)
    print(output1)


if __name__ == '__main__':
    args = parse_args()
    if args.convert_type == "dcp2hf":
        gemini80b_transform_dcp2hf(args.input, args.output)
    elif args.convert_type == "hf2dcp":
        gemini80b_transform_hf2dcp(args.input, args.output)
    else:
        raise ValueError(f"Unsupported convert type: {args.convert_type}")
