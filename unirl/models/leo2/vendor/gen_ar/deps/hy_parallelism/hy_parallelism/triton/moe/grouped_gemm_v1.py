import torch
from hy_parallelism.models.modules.moe.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk, group_gemm_same_mn

def triton_grouped_mm(x, w, num_tokens_per_expert):
    """
    m,k  e,k,n -> m,n

    m: num token
    k: input_dim
    e: num experts
    n: output_dim

    对标 torch._grouped_mm
    """
    x = x.contiguous()
    w = w.contiguous()
    # assert w.shape[1] == w.shape[2], f"w.shape = {w.shape}"
    assert x.shape[1] == w.shape[1], f"x.shape = {x.shape}, w.shape = {w.shape}"
    return group_gemm_same_nk(
        a=x,
        b=w,
        cumsum_M=torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32),
        max_M=x.shape[0],
        transpose_a=False,
        transpose_b=False,
    )


def triton_grouped_mm_given_offs(x, w, offs):
    x = x.contiguous()
    w = w.contiguous()
    assert x.shape[1] == w.shape[1], f"x.shape = {x.shape}, w.shape = {w.shape}"
    return group_gemm_same_nk(
        a=x,
        b=w,
        cumsum_M=offs,
        max_M=x.shape[0],
        transpose_a=False,
        transpose_b=False,
    )


def naive_grouped_mm(x, w, num_tokens_per_expert):
    """
    m,k  e,k,n -> m,n

    m: num token
    k: input_dim
    e: num experts
    n: output_dim
    """
    out_chunks = []
    start = 0
    for i, n_tok in enumerate(num_tokens_per_expert.tolist()):
        end = start + n_tok
        in_chunk = x[start:end]  # [n_tok, hidden_dim]
        out_chunk = in_chunk @ w[i]  # [n_tok, hidden_dim]
        out_chunks.append(out_chunk)
        start = end
    return torch.cat(out_chunks, dim=0)

class GroupedMM(torch.autograd.Function):
    
    @staticmethod
    def forward(x, w, offs):
        """
        Forward pass: x (m, k) @ w (e, k, n) -> output (m, n)
        
        Args:
            x: input tensor of shape (m, k) where m is total tokens, k is input_dim
            w: weight tensor of shape (e, k, n) where e is num_experts, k is input_dim, n is output_dim
            offs: cumulative sum of tokens per expert, shape (e,)
        """
        out = triton_grouped_mm_given_offs(x, w, offs)
        return out
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up the context for backward and JVP passes.
        
        Args:
            ctx: The context object
            inputs: Tuple of (x, w, offs)
            output: The output tensor from forward pass
        """
        x, w, offs = inputs
        ctx.x = x
        ctx.w = w
        ctx.offs = offs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass:
        - grad_x: grad_output (m, n) @ w.transpose (e, n, k) -> grad_x (m, k)
        - grad_w: x.transpose (m, k) @ grad_output (m, n) -> grad_w (e, k, n)
        
        Args:
            grad_output: gradient of output, shape (m, n)
        """
        x, w, offs = ctx.x, ctx.w, ctx.offs
        
        grad_x = None
        grad_w = None
        
        # Compute grad_x: grad_output (m, n) @ w.transpose (e, n, k) -> grad_x (m, k)
        if ctx.needs_input_grad[0]:
            # w needs to be transposed: (e, k, n) -> (e, n, k)
            w_transposed = w.transpose(-2, -1)  # (e, n, k)
            grad_x = group_gemm_same_nk(
                a=grad_output.contiguous(),
                b=w_transposed.contiguous(),
                cumsum_M=offs,
                max_M=grad_output.shape[0],
                transpose_a=False,
                transpose_b=False,
            )
        
        # Compute grad_w: x[i]^T (k, m_i) @ grad_output[i] (m_i, n) -> grad_w[i] (k, n)
        # group_gemm_same_mn computes: a^T (M, k_i) @ b (k_i, N) -> c[i] (M, N)
        # where a is (total_K, M), b is (total_K, N), c is (G, M, N)
        # We need: x[i]^T (k, m_i) @ grad_output[i] (m_i, n) -> grad_w[i] (k, n)
        # So: a=x (m, k) -> (total_K=m, M=k), b=grad_output (m, n) -> (total_K=m, N=n)
        # Then: a^T (k, m_i) @ b (m_i, n) -> c[i] (k, n) = grad_w[i] (k, n)
        if ctx.needs_input_grad[1]:
            # Initialize grad_w with same shape as w
            grad_w = torch.empty_like(w).contiguous()
            # group_gemm_same_mn: a (total_K, M) @ b (total_K, N) -> c (G, M, N)
            # We need: x^T (k, m_i) @ grad_output (m_i, n) -> grad_w[i] (k, n)
            # So: a=x (m, k) -> (total_K=m, M=k), b=grad_output (m, n) -> (total_K=m, N=n)
            group_gemm_same_mn(
                a=x.contiguous(),            # (m, k) -> (total_K=m, M=k)
                b=grad_output.contiguous(),  # (m, n) -> (total_K=m, N=n)
                c=grad_w,       # (e, k, n)
                cumsum_K=offs,
                max_K=x.shape[0],
                transpose_a=True,
                transpose_b=False,
            )
            # Wrong implementation:
            # group_gemm_same_mn(
            #     a=grad_output.contiguous(),
            #     b=x.contiguous(),  # (m, n) -> (total_K=m, N=n)
            #     c=grad_w,       # (e, k, n)
            #     cumsum_K=offs,
            #     max_K=x.shape[0],
            #     transpose_a=True,
            #     transpose_b=False,
            # )
        
        return grad_x, grad_w, None  # None for offs gradient
    
    @staticmethod
    def jvp(ctx, x_t, w_t, offs_t): # Not tested yet.
        """
        JVP (Jacobian-Vector Product) pass:
        - y_t = x_t @ w + x @ w_t
        
        Args:
            ctx: The context from forward pass
            x_t: Tangent vector for x, shape (m, k) or None
            w_t: Tangent vector for w, shape (e, k, n) or None
            offs_t: Tangent vector for offs (should be None, as offs is not differentiable)
        
        Returns:
            Tangent vector for output, shape (m, n)
        """
        x, w, offs = ctx.x, ctx.w, ctx.offs
        
        # Compute y_t = x_t @ w + x @ w_t
        # First term: x_t @ w
        y_t_term1 = None
        if x_t is not None:
            y_t_term1 = triton_grouped_mm_given_offs(x_t, w, offs)
        
        # Second term: x @ w_t
        y_t_term2 = None
        if w_t is not None:
            y_t_term2 = triton_grouped_mm_given_offs(x, w_t, offs)
        
        # Combine the terms
        if y_t_term1 is not None and y_t_term2 is not None:
            y_t = y_t_term1 + y_t_term2
        elif y_t_term1 is not None:
            y_t = y_t_term1
        elif y_t_term2 is not None:
            y_t = y_t_term2
        else:
            # Both tangents are None - compute output shape from forward pass
            # Output shape is (m, n) where m = x.shape[0], n = w.shape[2]
            m, n = x.shape[0], w.shape[2]
            y_t = torch.zeros(m, n, device=x.device, dtype=x.dtype)
        
        return y_t
    
