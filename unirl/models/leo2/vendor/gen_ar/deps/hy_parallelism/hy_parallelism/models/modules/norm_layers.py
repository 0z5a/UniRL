"""
RMS Normalization Triton Kernel
https://github.com/kapilsh/gpt-oss-scratch/blob/main/kernels/rms_norm.py
===============================

Optimized Triton implementation of RMS Normalization with forward and backward passes.

RMS Normalization computes:
y = (x / sqrt(mean(x^2) + eps)) * weight
"""

import torch
import triton
import torch.nn.functional as F
import triton.language as tl
import pandas as pd
from loguru import logger
from torch import nn
import matplotlib.pyplot as plt

# Configure torch settings for compiled functions
try:
    torch._functorch.config.donated_buffer = False  # type: ignore
except AttributeError:
    # Handle cases where _functorch is not available
    pass

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _rms_norm_fwd_fused(
    input_ptr,  # pointer to the input tensor
    output_ptr,  # pointer to the output tensor
    weight_ptr,  # pointer to the weight tensor
    rstd_ptr,  # pointer to the reciprocal standard deviation tensor
    row_stride,  # stride for moving to the next row
    feature_dim,  # number of features (columns) in input
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of input and output tensors to compute
    row_idx = tl.program_id(0)
    row_offset = row_idx.to(tl.int64) * row_stride.to(tl.int64)
    output_ptr += row_offset
    input_ptr += row_offset

    # Compute variance (mean of squared values for RMS)
    sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_values = tl.load(
            input_ptr + col_indices, mask=col_indices < feature_dim, other=0.0
        ).to(tl.float32)
        sum_of_squares += input_values * input_values

    variance = tl.sum(sum_of_squares, axis=0) / feature_dim
    reciprocal_std = 1 / tl.sqrt(variance + eps)

    # Store reciprocal standard deviation for backward pass
    tl.store(rstd_ptr + row_idx, reciprocal_std)

    # Normalize input and apply weight transformation
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        valid_mask = col_indices < feature_dim

        weight_values = tl.load(weight_ptr + col_indices, mask=valid_mask)
        input_values = tl.load(input_ptr + col_indices, mask=valid_mask, other=0.0).to(
            tl.float32
        )

        normalized_values = input_values * reciprocal_std
        output_values = normalized_values * weight_values

        # Write final output
        tl.store(output_ptr + col_indices, output_values, mask=valid_mask)


@triton.jit
def _rms_norm_bwd_dx_fused(
    dx_ptr,  # pointer to the input gradient
    dy_ptr,  # pointer to the output gradient
    dw_ptr,  # pointer to the partial sum of weights gradient
    input_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    rstd_ptr,  # pointer to the reciprocal standard deviation
    lock_ptr,  # pointer to the lock
    row_stride,  # stride for moving to the next row
    feature_dim,  # number of features (columns) in input
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_offset = row_idx.to(tl.int64) * row_stride.to(tl.int64)
    input_ptr += row_offset
    dy_ptr += row_offset
    dx_ptr += row_offset

    lock_id = row_idx % GROUP_SIZE_M
    lock_ptr += lock_id
    count_ptr = lock_ptr + GROUP_SIZE_M
    dw_row_ptr = dw_ptr + lock_id * feature_dim

    reciprocal_std = tl.load(rstd_ptr + row_idx)

    # Sum correction term over all feature columns
    correction = tl.zeros([], dtype=tl.float32)
    for block_offset in range(0, feature_dim, BLOCK_SIZE_N):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE_N)
        valid_mask = col_indices < feature_dim
        input_values = tl.load(input_ptr + col_indices, mask=valid_mask, other=0).to(
            tl.float32
        )
        dy_values = tl.load(dy_ptr + col_indices, mask=valid_mask, other=0).to(
            tl.float32
        )
        weight_values = tl.load(weight_ptr + col_indices, mask=valid_mask).to(tl.float32)
        normalized_values = input_values * reciprocal_std
        weight_dy = weight_values * dy_values
        normalized_masked = tl.where(valid_mask, normalized_values, 0.0)
        weight_dy_masked = tl.where(valid_mask, weight_dy, 0.0)
        correction += tl.sum(
            weight_dy_masked * normalized_masked * input_values, axis=0
        )
    correction = correction / feature_dim

    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass
    count = tl.load(count_ptr)
    for block_offset in range(0, feature_dim, BLOCK_SIZE_N):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE_N)
        valid_mask = col_indices < feature_dim
        input_values = tl.load(input_ptr + col_indices, mask=valid_mask, other=0).to(
            tl.float32
        )
        dy_values = tl.load(dy_ptr + col_indices, mask=valid_mask, other=0).to(
            tl.float32
        )
        weight_values = tl.load(weight_ptr + col_indices, mask=valid_mask).to(tl.float32)
        normalized_values = input_values * reciprocal_std
        weight_dy = weight_values * dy_values
        weight_dy_masked = tl.where(valid_mask, weight_dy, 0.0)
        dx_values = (
            weight_dy_masked - input_values * reciprocal_std * correction
        ) * reciprocal_std
        tl.store(dx_ptr + col_indices, dx_values, mask=valid_mask)

        partial_dw = (dy_values * normalized_values).to(weight_values.dtype)
        dw_col_ptr = dw_row_ptr + col_indices
        if count == 0:
            pass
        else:
            partial_dw += tl.load(dw_col_ptr, mask=valid_mask)
        tl.store(dw_col_ptr, partial_dw, mask=valid_mask)

    if count == 0:
        tl.atomic_xchg(count_ptr, 1)
    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0)


class RMSNorm(torch.autograd.Function):
    """
    Triton-optimized RMS Normalization with automatic differentiation support.
    """

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        # Allocate output tensor
        if weight.shape != normalized_shape:
            raise RuntimeError(f"Expected weight to be of same shape as normalized_shape, but got weight of shape {weight.shape} and normalized_shape = {normalized_shape}")

        y = torch.empty_like(x)

        # Reshape input to 2D for processing
        x_reshaped = x.reshape(-1, x.shape[-1])
        batch_size, feature_dim = x_reshaped.shape
        rstd = torch.empty((batch_size,), dtype=torch.float32, device=x.device)

        # Determine optimal block size (limited by 64KB per feature)
        max_fused_size = 65536 // x.element_size()
        BLOCK_SIZE = min(max_fused_size, triton.next_power_of_2(feature_dim))

        if feature_dim > BLOCK_SIZE:
            raise RuntimeError("This RMS norm doesn't support feature dim >= 64KB.")

        # Heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # Launch forward kernel
        _rms_norm_fwd_fused[(batch_size,)](  # type: ignore
            x_reshaped,
            y,
            weight,
            rstd,
            x_reshaped.stride(0),
            feature_dim,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,  # type: ignore
        )

        # Save tensors for backward pass
        ctx.save_for_backward(x, weight, rstd)
        ctx.block_size = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        feature_dim = weight.shape[0]

        # Heuristics for parallel reduction group size
        if feature_dim <= 1024:
            group_size = 256
        elif feature_dim <= 4096:
            group_size = 128
        elif feature_dim <= 8192:
            group_size = 96
        else:
            group_size = 64

        # Allocate output tensors
        locks = torch.zeros(2 * group_size, dtype=torch.int32, device=weight.device)
        dw_partial = torch.zeros(
            (group_size, feature_dim), dtype=x.dtype, device=weight.device
        )
        # Reshape inputs for processing
        x_reshaped = x.reshape(-1, x.shape[-1]).contiguous()
        dy_reshaped = dy.reshape(-1, dy.shape[-1]).contiguous()
        batch_size, feature_dim = x_reshaped.shape
        dx_reshaped = torch.empty_like(x_reshaped)

        block_size_n = min(128, triton.next_power_of_2(feature_dim))

        # Launch backward kernel for input gradients and partial weight gradients
        _rms_norm_bwd_dx_fused[(batch_size,)](  # type: ignore
            dx_reshaped,
            dy_reshaped,
            dw_partial,
            x_reshaped,
            weight,
            rstd,
            locks,
            x_reshaped.stride(0),
            feature_dim,
            BLOCK_SIZE_N=block_size_n,  # type: ignore
            GROUP_SIZE_M=group_size,  # type: ignore
        )

        dw_final = dw_partial.sum(dim=0).to(weight.dtype)

        dx = dx_reshaped.view_as(dy)
        return dx, None, dw_final, None


# Convenience function for using the optimized RMS norm
rms_norm = RMSNorm.apply


def pytorch_rms_norm(x, weight, eps=1e-5):
    """PyTorch reference implementation of RMS Norm."""
    return F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)


# Compiled version for performance comparison
pytorch_rms_norm_compiled = torch.compile(pytorch_rms_norm)


def test_rms_norm_correctness(
    batch_size=1151, feature_dim=8192, dtype=torch.float16, eps=1e-5, device=None
):
    """Test correctness of Triton RMS Norm vs PyTorch implementation."""
    if device is None:
        device = DEVICE

    # Create test data
    weight = torch.rand(feature_dim, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(batch_size, feature_dim, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)

    # Forward pass comparison
    y_triton: torch.Tensor = rms_norm(x, (feature_dim,), weight, eps)  # type: ignore
    y_pytorch = pytorch_rms_norm(x, weight, eps)
    y_compiled = pytorch_rms_norm_compiled(x, weight, eps)

    # Backward pass - Triton
    y_triton.backward(dy, retain_graph=True)  # type: ignore
    assert (
        x.grad is not None and weight.grad is not None
    ), "Gradients should be computed"
    dx_triton = x.grad.clone()
    dw_triton = weight.grad.clone()
    x.grad, weight.grad = None, None

    # Backward pass - PyTorch
    y_pytorch.backward(dy, retain_graph=True)
    assert (
        x.grad is not None and weight.grad is not None
    ), "Gradients should be computed"
    dx_pytorch = x.grad.clone()
    dw_pytorch = weight.grad.clone()
    x.grad, weight.grad = None, None

    # Backward pass - Compiled PyTorch
    y_compiled.backward(dy, retain_graph=True)
    assert (
        x.grad is not None and weight.grad is not None
    ), "Gradients should be computed"
    dx_compiled = x.grad.clone()
    dw_compiled = weight.grad.clone()

    # Assertions
    assert torch.allclose(
        y_triton, y_pytorch, atol=1e-2, rtol=0
    ), "Forward pass mismatch: Triton vs PyTorch"
    assert torch.allclose(
        y_triton, y_compiled, atol=1e-2, rtol=0
    ), "Forward pass mismatch: Triton vs Compiled"
    assert torch.allclose(
        dx_triton, dx_pytorch, atol=1e-2, rtol=0
    ), "Input grad mismatch: Triton vs PyTorch"
    assert torch.allclose(
        dx_triton, dx_compiled, atol=1e-2, rtol=0
    ), "Input grad mismatch: Triton vs Compiled"
    assert torch.allclose(
        dw_triton, dw_pytorch, atol=1e-2, rtol=0
    ), "Weight grad mismatch: Triton vs PyTorch"
    assert torch.allclose(
        dw_triton, dw_compiled, atol=1e-2, rtol=0
    ), "Weight grad mismatch: Triton vs Compiled"

    logger.success("All correctness tests passed!")

class HunyuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        """
        HunyuanRMSNorm is equivalent to T5LayerNorm
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.variance_epsilon = eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class FusedHunyuanRMSNorm(HunyuanRMSNorm):

    def forward(self, hidden_states):
        from liger_kernel.transformers import LigerRMSNorm
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        return LigerRMSNormFunction.apply(hidden_states, self.weight, self.variance_epsilon, 0.0, "llama", False, None)
        return rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon)

class NativeHunyuanRMSNorm(HunyuanRMSNorm):

    def forward(self, hidden_states):
        return F.rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon)


class TritonOnlyHunyuanRMSNorm(HunyuanRMSNorm):
    def forward(self, hidden_states):
        return rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon)

class TritonHunyuanRMSNorm(HunyuanRMSNorm):

    def forward(self, hidden_states):
        if self.training or hidden_states.requires_grad:
            return F.rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon)
        else:
            return rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon)

def test_fused_hunyuan_rms_norm_correctness():
    hidden_size = 3152
    eps = 1e-5
    device = DEVICE
    dtype = torch.float32
    hidden_states = torch.randn(1, 200000, hidden_size, dtype=dtype, device=device)
    fused_hunyuan_rms_norm = FusedHunyuanRMSNorm(hidden_size, eps, device, dtype)
    hunyuan_rms_norm = NativeHunyuanRMSNorm(hidden_size, eps, device, dtype)
    triton_hunyuan_rms_norm = TritonOnlyHunyuanRMSNorm(hidden_size, eps, device, dtype)
    hunyuan_rms_norm.load_state_dict(fused_hunyuan_rms_norm.state_dict())
    triton_hunyuan_rms_norm.load_state_dict(fused_hunyuan_rms_norm.state_dict())
    from hy_parallelism.bing_utils import Timer
    times = 10
    with Timer("native_hunyuan_rms_norm"):
        for _ in range(times):
            out2 = hunyuan_rms_norm(hidden_states)
    with Timer("fused_hunyuan_rms_norm"):
        for _ in range(times):
            out1 = fused_hunyuan_rms_norm(hidden_states)
    with Timer("triton_hunyuan_rms_norm"):
        for _ in range(times):
            out3 = triton_hunyuan_rms_norm(hidden_states)
    out1.sum().backward()
    out2.sum().backward()
    out3.sum().backward()
    dic1 = {
        'out': out1,
        'grad': hunyuan_rms_norm.weight.grad,
    }
    dic2 = {
        'out': out2,
        'grad': hunyuan_rms_norm.weight.grad,
    }
    dic3 = {
        'out': out3,
        'grad': triton_hunyuan_rms_norm.weight.grad,
    }
    from hy_parallelism.tools.precision_aligner import tensor_precision_report
    report = tensor_precision_report(dic1, dic2)
    report = tensor_precision_report(dic1, dic3)
    print(report)



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i if i < 9 else 512 * i for i in range(1, 54)],
        line_arg="provider",
        line_vals=["triton", "torch", "torch_compile"],
        line_names=["Triton", "Torch", "Torch Compile"],
        styles=[
            ("#1f77b4", "-", "o"),
            ("#2ca02c", "-.", "^"),
            ("#d62728", "--", "s"),
        ],
        ylabel="GB/s",
        plot_name="rms-norm-forward",
        args={"M": 4096, "dtype": torch.float16, "mode": "forward"},
    )
)
def bench_rms_norm_forward(
    M, N, dtype, provider, mode="forward", eps=1e-5, device=None
):
    """Benchmark RMS Norm forward pass."""
    if device is None:
        device = DEVICE

    # Create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return rms_norm(x, w_shape, weight, eps)
        if provider == "torch":
            return pytorch_rms_norm(x, weight, eps)
        if provider == "torch_compile":
            return pytorch_rms_norm_compiled(x, weight, eps)

    # Forward pass benchmark
    if mode == "forward":
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(  # type: ignore
            y_fwd, quantiles=quantiles, rep=500, return_mode="all"
        )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["triton", "torch", "torch_compile"],
        line_names=["Triton", "Torch", "Torch Compile"],
        styles=[
            ("#1f77b4", "-"),
            ("#2ca02c", "-."),
            ("#d62728", "--"),
        ],
        ylabel="GB/s",
        plot_name="rms-norm-backward",
        args={"M": 4096, "dtype": torch.float16, "mode": "backward"},
    )
)


def bench_rms_norm_backward(
    M, N, dtype, provider, mode="backward", eps=1e-5, device=None
):
    """Benchmark RMS Norm backward pass."""
    if device is None:
        device = DEVICE

    # Create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return rms_norm(x, w_shape, weight, eps)
        if provider == "torch":
            return pytorch_rms_norm(x, weight, eps)
        if provider == "torch_compile":
            return pytorch_rms_norm_compiled(x, weight, eps)

    # Backward pass benchmark
    if mode == "backward":

        def backward_fn():
            # Clear gradients first
            if x.grad is not None:
                x.grad = None
            if weight.grad is not None:
                weight.grad = None

            # Get fresh forward pass
            y = y_fwd()

            # For compiled functions, we need create_graph=False, retain_graph=False
            if provider == "torch_compile":
                y.backward(dy, create_graph=False, retain_graph=False)  # type: ignore
            else:
                y.backward(dy, retain_graph=True)  # type: ignore

        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(  # type: ignore
            backward_fn,
            quantiles=quantiles,
            grad_to_none=[x, weight],
            rep=500,
        )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


def run_performance_benchmarks(
    print_data=True,
    include_backward=False,
):
    """Run comprehensive performance benchmarks for RMS Norm."""
    logger.info("Starting RMS Norm performance benchmarks...")

    try:
        logger.info("Running forward pass benchmark...")
        forward_results = bench_rms_norm_forward.run(
            print_data=print_data, return_df=True, show_plots=True
        )
        logger.success("Forward pass benchmark completed successfully!")

        backward_results = None
        if include_backward:
            try:
                logger.info("Running backward pass benchmark...")
                backward_results = bench_rms_norm_backward.run(
                    print_data=print_data, return_df=True, show_plots=True
                )
                logger.success("Backward pass benchmark completed successfully!")
            except Exception as e:
                logger.warning(f"Backward benchmark failed: {e}")
                logger.info("Continuing with forward benchmark only...")

        logger.success("Performance benchmarks completed successfully!")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise