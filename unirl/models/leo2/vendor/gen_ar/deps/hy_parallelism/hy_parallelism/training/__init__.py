from functools import cache
from hy_parallelism.training.activation_offload import OffloadActivations, get_activation_offload_context
from hy_parallelism.training.activation_offload_notrack import OffloadActivationsNotrack, get_activation_offload_context_notrack




if __name__ == "__main__":
    import torch
    model = torch.nn.Linear(10, 10)

    with get_activation_offload_context():
        model(torch.randn(10)).sum().backward() 
    with get_activation_offload_context_notrack():
        model(torch.randn(10)).sum().backward()

    
    with get_activation_offload_context():
        model(torch.randn(10)).sum()
    with get_activation_offload_context_notrack():
        model(torch.randn(10)).sum().backward()
