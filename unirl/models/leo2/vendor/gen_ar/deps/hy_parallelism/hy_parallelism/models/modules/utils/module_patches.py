from hy_parallelism.models.modules.moe import TokenChoiceTopKRouter
from hy_parallelism.models.modules.moe.routers.router_with_aux_loss import RouterWithAuxLoss
from . import replace_module

def replace_router_with_aux_loss_router(model):
    def get_new_router(full_name, child: TokenChoiceTopKRouter):
        new_router = RouterWithAuxLoss(
        )
        new_router.load_state_dict(child.state_dict())
        return new_router

    replace_module(
        model,
        is_target_module=lambda full_name, child: isinstance(child, TokenChoiceTopKRouter),
        get_alternative=get_new_router,
    )