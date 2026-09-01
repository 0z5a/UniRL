from typing import Dict, Any

import torch
import torch.nn as nn

from ...utils.torch_utils import PRECISION_TO_TYPE


class GptNeoxMLP(nn.Module):
    def __init__(self, config, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.mlp_bias, **factory_kwargs)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.mlp_bias, **factory_kwargs)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.mlp_bias, **factory_kwargs)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.mlp_bias, **factory_kwargs)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.mlp_bias, **factory_kwargs)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False, **factory_kwargs)
        self.experts = nn.ModuleList(LLaMAMLP(config, **factory_kwargs) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


class HunYuanMLP(nn.Module):
    def __init__(self, config, block_idx = None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.mlp_bias, **factory_kwargs)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.mlp_bias, **factory_kwargs)
        # self.gate_proj_bias = nn.Parameter(self.gate_proj.bias)
        # self.up_proj_bias = nn.Parameter(self.up_proj.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.mlp_bias, **factory_kwargs)

    def forward(self, x):
        gate_proj = self.gate_proj(x)
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(torch.nn.functional.silu(gate_proj) * up_proj)
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LightProjector(nn.Module):
    def __init__(self, projector_type, input_dim, n_embed, depth=1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if projector_type == "linear":
            modules = nn.Linear(input_dim, n_embed, **factory_kwargs)

        elif projector_type == "mlp_gelu":
            modules = [nn.Linear(input_dim, n_embed, **factory_kwargs)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed, n_embed, **factory_kwargs))
            modules = nn.Sequential(*modules)
        
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
    
        self.layers = modules

    def forward(self, x):
        return self.layers(x)


def load_projector(
        projector_type: str | None = None,
        projector_precision: str | None = None,
        projector_params: Dict[str, Any] | None = None,
        device=None,
        require_grad=True,
        eval_mode=False,
        config=None,
):
    if config is not None:
        projector_type = config["vision_aligner_type"]
        projector_precision = config["vision_aligner_precision"]
        projector_params = config["vision_aligner_params"]
        require_grad = not config.get("vision_model_freeze", not require_grad)
        eval_mode = config.get("vision_model_freeze", eval_mode)

    projector = LightProjector(
        projector_type=projector_type,
        **projector_params,
    )

    if projector_precision is not None:
        projector = projector.to(dtype=PRECISION_TO_TYPE[projector_precision])

    if device is not None:
        projector = projector.to(device=device)

    if not require_grad:
        projector.requires_grad_(False)

    if eval_mode:
        projector.eval()

    return projector

class NaiveMLP(nn.Module):
    def __init__(self, mlp_depth, input_dim, inter_dim, out_dim, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        modules = [nn.Linear(input_dim, inter_dim, **factory_kwargs)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(inter_dim, out_dim, **factory_kwargs))
        modules = nn.Sequential(*modules)

        self.layers = modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
