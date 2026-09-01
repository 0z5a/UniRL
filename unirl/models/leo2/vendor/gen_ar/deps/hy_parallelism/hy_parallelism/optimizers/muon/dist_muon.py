import os
import math
import torch
import torch.distributed as dist
import torch.distributed
try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    )
    from torch.distributed.tensor._api  import  _ToTorchTensor
except ImportError:
    # handle old pytorch versions
    Dtensor = None

# This code is modified from the GitHub repository of KellerJordan:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    # X = G.bfloat16()
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Dist_Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
        dist_size: the scale of the distributed muon.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        fsdp_strategy='none',
        dist_size=8
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim >= 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

        self.dist_size = dist_size
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        assert self.world_size % self.dist_size == 0, "Error: self.world_size %% self.dist_size != 0" 
        self.local_rank = self.rank % self.dist_size
        self.pg_groups = []
        for i in range(self.world_size // self.dist_size):
            offset = i * self.dist_size
            ranks = [offset + local_i for local_i in range(self.dist_size)]
            self.pg_groups.append(torch.distributed.new_group(ranks))
        
    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################
            dist_size = self.dist_size
            rank = self.rank
            local_rank = self.local_rank

            params = [ p for p in group["params"] if self.state[p]["use_muon"] ]
            padding = torch.empty_like(params[-1])
            padding.grad = torch.zeros_like(params[-1].grad)    
            params_pad = params + [padding] * (dist_size - len(params) % dist_size)
            
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for base_i in range(len(params))[::dist_size]:
                g_buf = []
                device_mesh_list=[]
                for i, p in enumerate(params_pad[base_i: base_i+dist_size]):
                    if p.grad is None:
                        raise ValueError(f"Gradient of some parameters is None")
                    if isinstance(p.grad, DTensor):
                        device_mesh_list.append(p.grad.device_mesh)
                        g_buf.append(p.grad.full_tensor().bfloat16())
                    else:
                        device_mesh_list.append(None)
                        g_buf.append(p.grad)
                if base_i+local_rank < len(params):
                    g = g_buf[local_rank]
                    p = params_pad[base_i: base_i+dist_size][local_rank]
                    # sanity check
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None

                    # calc update
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    
                    #update
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    g_buf[local_rank] = g
                
                #allgather update
                group_idx = rank // dist_size
                dist.all_gather(g_buf, g_buf[local_rank], group=self.pg_groups[group_idx])
                    
                for i in range(dist_size):
                    p = params_pad[base_i + i]
                    g = g_buf[i]
                    device_mesh = device_mesh_list[i]
                    if device_mesh != None:
                        u = DTensor.from_local(g, device_mesh)
                    else:
                        u = g    
                    # scale update
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                    
                    # apply weight decay
                    p.data.mul_(1 - lr * wd)
                    
                    # apply update
                    p.data.add_(u.view(p.shape), alpha=-adjusted_lr)

                
            ############################
            #       AdamW backup       #
            ############################
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    if isinstance(p, DTensor):
                        g = DTensor.from_local(torch.zeros_like(p.to_local()), p.device_mesh, placements=p.placements)
                    else:
                        g = torch.zeros_like(p)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
        
        return loss


def get_dist_muon_optimizer(model, lr=1e-3,  weight_decay=0.1, momentum=0.95, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, dist_size=8):
    muon_params = [
        p
        for name, p in model.named_parameters()
        if p.ndim >= 2 
        #if p.ndim == 2 and "embed_tokens" not in name and "final_layer" not in name
    ]
    adamw_params = [
        p
        for name, p in model.named_parameters()
        if not (
            p.ndim >= 2
            #p.ndim == 2 and "embed_tokens" not in name and "final_layer" not in name
        )
    ]

    return Dist_Muon(
        lr=lr,
        wd=weight_decay,
        muon_params=muon_params,
        momentum=momentum,
        adamw_params=adamw_params,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
        dist_size=dist_size
    )

def get_dist_muon_for_engine(params, lr=1e-3,  weight_decay=0.1, momentum=0.95, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, dist_size=8):
    muon_params = [
        p
        for p in params
        if p.ndim >= 2
    ]
    adamw_params = [
        p
        for p in params
        if not (p.ndim >= 2)
    ]
    return Dist_Muon(
        lr=lr,
        wd=weight_decay,
        muon_params=muon_params,
        momentum=momentum,
        adamw_params=adamw_params,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
        dist_size=dist_size,
    )

def get_dist_muon_for_engine_using_all_params(muon_params, lr=1e-3,  weight_decay=0.1, momentum=0.95, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, dist_size=8):
    return Dist_Muon(
        lr=lr,
        wd=weight_decay,
        muon_params=muon_params,
        momentum=momentum,
        adamw_params=[],
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
        dist_size=dist_size,
    )