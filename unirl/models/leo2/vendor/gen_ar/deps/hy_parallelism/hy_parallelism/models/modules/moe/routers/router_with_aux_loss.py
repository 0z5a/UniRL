"""
Router with auxiliary losses (balance loss and z-loss) for MoE.

This module extends TokenChoiceTopKRouter from torchtitan to compute
balance loss and router z-loss similar to the implementation in hunyuan_moe_torch.py.
"""

from typing import Literal

import loguru
import torch
import torch.nn.functional as F

try:
    from torchtitan.models.moe.moe import TokenChoiceTopKRouter
except ImportError:
    raise ImportError(
        "torchtitan is required. Please install it or ensure it's available in your environment."
    )


class RouterWithAuxLoss(TokenChoiceTopKRouter):
    """Router that extends TokenChoiceTopKRouter with balance loss and z-loss computation.
    
    This router computes two auxiliary losses:
    1. Balance loss: Encourages balanced token distribution across experts
    2. Router z-loss: Regularizes router logits to prevent extreme values
    
    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        num_expert_groups (int | None): Number of expert groups for node-limited routing. 
            If None, standard top-k routing is used. Must be a divisor of num_experts.
        num_limited_groups (int | None): Number of groups to select in node-limited routing. 
            Required when num_expert_groups is set.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid"]): Whether to use sigmoid or softmax for router scores.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
        return_loss (bool): Whether to compute and return auxiliary losses. Defaults to True.
        _debug_force_load_balance (bool): Debug flag for balanced round-robin routing.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_expert_groups: int | None = None,
        num_limited_groups: int | None = None,
        top_k: int = 2,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_norm: bool = False,
        route_scale: float = 1.0,
        return_loss: bool = True,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__(
            dim=dim,
            num_experts=num_experts,
            num_expert_groups=num_expert_groups,
            num_limited_groups=num_limited_groups,
            top_k=top_k,
            score_func=score_func,
            route_norm=route_norm,
            route_scale=route_scale,
            _debug_force_load_balance=_debug_force_load_balance,
        )
        self.return_loss = return_loss

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass with auxiliary loss computation.
        
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
                - balance_loss (torch.Tensor | None):
                    Balance loss scalar. None if return_loss is False.
                - router_z_loss (torch.Tensor | None):
                    Router z-loss scalar. None if return_loss is False.
        """
        # Get raw logits before softmax/sigmoid
        raw_logits = self.gate(x)  # shape: (bs*slen, num_experts)
        
        # Call parent forward to get routing results
        top_scores, selected_experts_indices, num_tokens_per_expert = super().forward(x, expert_bias)
        
        balance_loss = None
        router_z_loss = None
        
        if self.return_loss:
            # Compute balance loss
            # Similar to topkgating_impl2: use top-2 mask for balance loss
            
            # Create one-hot masks for top-k experts
            # For top-2 balance loss, we use top-1 and top-2 selections
            if self.top_k >= 2:
                # Get top-1 and top-2 expert indices
                top1_indices = selected_experts_indices[:, 0]  # shape: (n_tokens,)
                top2_indices = selected_experts_indices[:, 1]  # shape: (n_tokens,)
                
                # Create one-hot masks
                top1_mask = F.one_hot(top1_indices, num_classes=self.num_experts)  # shape: (n_tokens, num_experts)
                top2_mask = F.one_hot(top2_indices, num_classes=self.num_experts)  # shape: (n_tokens, num_experts)
                
                # Combine top-1 and top-2 masks (union)
                combined_mask = (top1_mask | top2_mask).float()  # shape: (n_tokens, num_experts)
                
                # Compute density_1: mean of combined mask across tokens
                density_1 = combined_mask.mean(dim=0)  # shape: (num_experts,)
            else:
                # If top_k < 2, use only top-1
                top1_indices = selected_experts_indices[:, 0]
                top1_mask = F.one_hot(top1_indices, num_classes=self.num_experts).float()
                density_1 = top1_mask.mean(dim=0)  # shape: (num_experts,)
            
            # Compute density_1_proxy from raw gates (softmax/sigmoid probabilities)
            # Recompute gates to get probabilities
            if self.score_func == "sigmoid":
                raw_gates = torch.sigmoid(raw_logits.to(torch.float32))
            elif self.score_func == "softmax":
                raw_gates = F.softmax(raw_logits.to(torch.float32), dim=1)
            else:
                raise NotImplementedError(f"Unknown score function {self.score_func}")
            
            # Compute mean probability per expert
            density_1_proxy = raw_gates.mean(dim=0)  # shape: (num_experts,)
            
            # Balance loss: encourage balanced distribution
            balance_loss = (density_1_proxy * density_1).mean() * float(self.num_experts ** 2)
            
            # Compute router z-loss
            # z-loss = mean(square(logsumexp(logits)))
            router_z_loss_tmp = torch.logsumexp(raw_logits, dim=-1)  # shape: (n_tokens,)
            router_z_loss_tmp = torch.square(router_z_loss_tmp)  # shape: (n_tokens,)
            router_z_loss = router_z_loss_tmp.mean()  # scalar
        
        aux_loss = balance_loss + router_z_loss
        from hy_parallelism.training.utils import attatch_loss_to_activation
        top_scores = attatch_loss_to_activation(top_scores, aux_loss)
        # loguru.logger.info(f'aux_loss: {aux_loss=} {balance_loss=} {router_z_loss=}')
        # return top_scores, selected_experts_indices, num_tokens_per_expert, balance_loss, router_z_loss
        return top_scores, selected_experts_indices, num_tokens_per_expert

