from hy_parallelism.models.modules.moe import MoE, MoEArgs
from hy_parallelism.models.modules.moe.routers.router_with_aux_loss import RouterWithAuxLoss

class MOEWithAuxLoss(MoE):

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args, dim, hidden_dim)

        num_experts = moe_args.num_experts
        self.router = RouterWithAuxLoss(
            dim=dim,
            num_experts=num_experts,
            num_expert_groups=moe_args.num_expert_groups,
            num_limited_groups=moe_args.num_limited_groups,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
            _debug_force_load_balance=moe_args._debug_force_load_balance,
        )

    