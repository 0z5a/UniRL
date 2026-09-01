# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
from typing import Dict, Any
import math
import einops

import torch
import torch.nn as nn
from loguru import logger
from ...utils.torch_utils import PRECISION_TO_TYPE

# FFN
def FeedForward(dim, mult=4, dtype=None, device=None):
    factory_kwargs = {'dtype': dtype, 'device': device}
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim, **factory_kwargs),
        nn.Linear(dim, inner_dim, bias=False, **factory_kwargs),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False, **factory_kwargs),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x



class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, dtype=None, device=None):
        super().__init__()
        factory_kwargs = {'dtype': dtype, 'device': device}
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim, **factory_kwargs)
        self.norm2 = nn.LayerNorm(dim, **factory_kwargs)

        self.to_q = nn.Linear(dim, inner_dim, bias=False, **factory_kwargs)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False, **factory_kwargs)
        self.to_out = nn.Linear(inner_dim, dim, bias=False, **factory_kwargs)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """

        x = self.norm1(x) # [2, 1, 1280]
        latents = self.norm2(latents) # torch.Size([2, 16, 1280])
        
        b, l, _ = latents.shape

        q = self.to_q(latents) # [2, 16, 1280]
        kv_input = torch.cat((x, latents), dim=-2) # [2, 1+16, 1280]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1) # [2, 1+16, 2560]
        
        q = reshape_tensor(q, self.heads) # b, n_heads, length, dim_per_head ([2, 20, 16, 64])
        k = reshape_tensor(k, self.heads) # b, n_heads, length, dim_per_head ([2, 20, 1+16, 64])
        v = reshape_tensor(v, self.heads) # b, n_heads, length, dim_per_head ([2, 20, 1+16, 64])

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards ;torch.Size([2, 20, 16, 1+16])
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # visualize weight torchvision.utils.save_image(weight, "weight.png")
        # import torchvision
        # torchvision.utils.save_image(weight[0, 0], "weight.png")
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=2048,
        depth=4,
        dim_head=64,
        heads=32,
        input_token=1,
        num_queries=16,
        embedding_dim=512,
        output_dim=2048,
        ff_mult=4,
        dtype=None,
        device=None,
        input_latents_type="cross"
    ):
        super().__init__()
        factory_kwargs = {'dtype': dtype, 'device': device}
        self.input_token = input_token
        self.num_queries = num_queries
        self.input_latents_type = input_latents_type
        if input_latents_type == "cross":
            self.latents = nn.Parameter(torch.randn(1, num_queries, dim, **factory_kwargs) / dim**0.5)
        elif input_latents_type == "self":
            self.latents_projection = nn.Linear(embedding_dim*input_token, dim*num_queries, **factory_kwargs)
        else:
            raise ValueError(f"Invalid input_latents_type: {input_latents_type}")

        
        self.proj_in = nn.Linear(embedding_dim, dim, **factory_kwargs)

        self.proj_out = nn.Linear(dim, output_dim, **factory_kwargs)
        self.norm_out = nn.LayerNorm(output_dim, **factory_kwargs)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if x.ndim == 4:
            x = einops.rearrange(x, 'b c h w -> b (h w) c')
        elif x.ndim == 2:
            x = einops.rearrange(x, 'b c -> b 1 c')
        elif x.ndim == 3:
            pass
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        if self.input_latents_type == "cross":
            latents = self.latents.repeat(x.size(0), 1, 1) # 1, 16, 1280 -> 2, 16, 1280
        elif self.input_latents_type == "self":
            B, L, D = x.shape
            x = x.reshape(B, 1, L*D)
            latents = self.latents_projection(x) # 2,1,512 -> 2,1,1280*16
            latents = latents.reshape(B, self.num_queries, -1)
        else:
            raise ValueError(f"Invalid input_latents_type: {self.input_latents_type}")
        x = self.proj_in(x) # 2,1,512 -> 2,1,1280
        # latents works as q, (x, latents) works as kv
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        dtype = latents.dtype
        return self.norm_out(latents).to(dtype)

def load_qformer(
        qformer_type: str,
        qformer_precision: str,
        qformer_params: Dict[str, Any],
        device=None,
        require_grad=True,
        eval_mode=False,
):
    projector = Resampler(
        input_latents_type=qformer_type,
        **qformer_params,
    )

    if qformer_precision is not None:
        projector = projector.to(dtype=PRECISION_TO_TYPE[qformer_precision])

    if device is not None:
        projector = projector.to(device=device)

    if not require_grad:
        projector.requires_grad_(False)

    if eval_mode:
        projector.eval()

    return projector



if __name__ == "__main__":
    # Test parameters
    batch_size = 2
    seq_length = 1
    embedding_dim = 512
    output_dim = 2048
    num_queries = 16
    
    # Create input tensor
    x = torch.randn(batch_size, seq_length, embedding_dim)
    
    # Initialize resampler
    resampler = Resampler(
        dim=2048,
        depth=4,
        dim_head=64,
        heads=32,
        num_queries=num_queries,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        ff_mult=4
    )
    
    # Run forward pass
    output = resampler(x)
    
    # Check output shape
    expected_shape = (batch_size, num_queries, output_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Check output values are valid (not NaN or inf)
    assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
    
    # print parameter number of resampler
    print(f"parameter number of resampler: {sum(p.numel() for p in resampler.parameters())}")
    # 200 M
    import ipdb; ipdb.set_trace()
    
    print("All tests passed!")
    # PYTHONPATH=. python3 hymm/models/autoregressive/resampler.py