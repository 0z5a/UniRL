from torch import nn
import torch
from fvcore.nn import FlopCountAnalysis
from hy_parallelism.context_parallel.attention import attention_no_sp


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(10, 10)
        self.to_k = nn.Linear(10, 10)
        self.to_v = nn.Linear(10, 10)
    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        return attention_no_sp(q, k, v)

if __name__ == "__main__":
    from hy_parallelism.parallel_states import init_parallel_state
    init_parallel_state()

    from tests.fixtures import TestTransformerModel
    model = TestTransformerModel(use_moe=True).cuda()
    input = torch.randn(10, 128).cuda()
    flops = FlopCountAnalysis(model, input)
    flops.total()

    print(flops.by_operator())
    print('='*100)
    print(flops.by_module())
    print('='*100)
    print(flops.by_module_and_operator())
    print('='*100)
    print(flops)
    from IPython import embed
    embed()