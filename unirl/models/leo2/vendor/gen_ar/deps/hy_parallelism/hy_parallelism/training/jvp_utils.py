import types
from packaging import version


import loguru
import torch
import contextlib
from contextlib import contextmanager


from torch._functorch.eager_transforms import grad_increment_nesting
from torch.autograd import forward_ad as fwAD

from hy_parallelism.engines.parallel_engine import BaseParallelEngine
from hy_parallelism.parallel_states import get_parallel_state

@contextlib.contextmanager
def jvp_guard():
    # HACK(kevinkhwu): 
    # See tolist issue in jvp: https://github.com/pytorch/pytorch/issues/161943
    # Under jvp guard, not entering jvp function...
    from torch._functorch.eager_transforms import enter_jvp_nesting, exit_jvp_nesting
    from torch._functorch.eager_transforms import JVP_NESTING
    for _ in range(JVP_NESTING):
        exit_jvp_nesting()
    yield
    for _ in range(JVP_NESTING):
        enter_jvp_nesting()
    raise DeprecationWarning("Exiting jvp guard will lead to incorrect jvp results.")


def tolist(tensor):
    from torch._functorch.eager_transforms import JVP_NESTING
    if JVP_NESTING == 0:
        return tensor.tolist()
    else:
        if tensor.ndim == 1:
            return [k.item() for k in tensor]
        else:
            assert tensor.ndim > 1
            return [tolist(k) for k in tensor]
        
def in_jvp_context():
    from torch._functorch.eager_transforms import JVP_NESTING
    return JVP_NESTING > 0

def torch_jvp_tolist_bug_is_fixed():
    """https://github.com/pytorch/pytorch/issues/171604"""

    import torch

    def test_tolist_with_grad():
        """Test to see if tolist works inside grad transformation."""

        def f(x):
            # inside grad, x is a GradTrackingTensor
            result = x.tolist()
            # tolist should return a python list and not fail
            assert isinstance(result, list)
            assert result == [1.0, 2.0, 3.0]
            return (x**2).sum()

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device='cuda')
        grad_f = torch.func.grad(f)
        result = grad_f(x)
        torch.testing.assert_close(result, torch.tensor([2.0, 4.0, 6.0]), check_device=False)

    def test_tolist2():
        import torch
        from torch._functorch.eager_transforms import jvp_increment_nesting

        with jvp_increment_nesting():
            torch.tensor([1, 2, 3, 4]).tolist()

    def test_jvp_with_async_collective_tensor():
        import torch
        from torch.distributed._functional_collectives import _maybe_wrap_tensor

        inp = torch.randn(1)

        class F(torch.autograd.Function):
            @staticmethod
            def forward(input):
                return input
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output
            @staticmethod
            def setup_context(ctx, inputs, output):
                pass
            @staticmethod
            def jvp(ctx, input_tangent):
                return _maybe_wrap_tensor(input_tangent)

        torch.func.jvp(F.apply, (inp,), (inp,))

    try:
        test_tolist_with_grad()
        test_tolist2()
        test_jvp_with_async_collective_tensor()
        return True
    except RuntimeError as e:
        if "doesn't have storage" in str(e):
            raise
            return False
        raise



def unwrap_tensor_for_jvp(obj):
    """
    Another implementation:


    from torch._C._functorch import get_unwrapped, is_functorch_wrapped_tensor

    # 检查是否是 functorch wrapped tensor
    if is_functorch_wrapped_tensor(tensor):
        # 直接 unwrap（适用于所有类型的 wrapper）
        unwrapped_tensor = get_unwrapped(tensor)
        return func(unwrapped_tensor)
    """
    from torch._C._functorch import current_level
    from torch._C._functorch import _unwrap_for_grad
    level = current_level()
    return _unwrap_for_grad(obj, level)

def safe_jvp_op(op):
    """`Function.apply` conducts proper unwrapping for jvp."""
    class MethodDescriptor:
        def __init__(self, func):
            self.func = func

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return types.MethodType(self.func, obj)

    class SafeJVPOp(torch.autograd.Function):
        @staticmethod
        def forward(*args, **kwargs):
            print(
                f"forward {op.__name__} with args {[type(k) for k in args]} and kwargs {kwargs}"
            )
            return op(*args, **kwargs)

        @staticmethod
        def setup_context(ctx, inputs, output):
            pass


    if isinstance(op, types.MethodDescriptorType):
        return MethodDescriptor(SafeJVPOp.apply)
    ret = SafeJVPOp.apply
    # safe_wait_tensor.__dict__['default'] = torch.ops._c10d_functional.wait_tensor.default
    # safe_wait_tensor.__dict__['__qualname__'] = torch.ops._c10d_functional.wait_tensor.__qualname__
    return ret


def reverse_mode_jvp(func, inputs, tangents):
    """
    https://math.stackexchange.com/questions/2195377/reverse-mode-differentiation-vs-forward-mode-differentiation-where-are-the-be/3119199#3119199

    Without creating the second order gradient graph for memory efficiency.
    """
    # return torch.autograd.functional.jvp(func, inputs, tangents, create_graph=True)
    with torch.enable_grad():
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not isinstance(tangents, tuple):
            tangents = (tangents,)
        
        inputs = tuple(inp.requires_grad_(True) if not inp.requires_grad else inp for inp in inputs)
        
        outputs = func(*inputs)
        
        is_outputs_tuple = isinstance(outputs, tuple)
        if not is_outputs_tuple:
            outputs = (outputs,)
        
        grad_outputs = tuple(
            torch.zeros_like(out, requires_grad=True) for out in outputs
        )
        
        grad_inputs = torch.autograd.grad(
            outputs, inputs, grad_outputs=grad_outputs,
            create_graph=True, allow_unused=False
        )
        
        grad_res = torch.autograd.grad(
            grad_inputs, grad_outputs, tangents,
            create_graph=False, allow_unused=False
        )
        if len(grad_res) == 1:
            jvp_output = grad_res[0]
        else:
            jvp_output = grad_res
        
        if not is_outputs_tuple:
            outputs = outputs[0]
    
    return outputs, jvp_output

def reverse_mode_jvp_element_wise(func, inputs, tangents):
    """ Wrong implementation, which only supports element-wise operations."""

    y = func(*inputs)
    
    if y.numel() == 1:
        grad_outputs = torch.tensor(1.0, device=y.device, dtype=y.dtype)
    else:
        grad_outputs = torch.ones_like(y)
    
    grads = torch.autograd.grad(
        y, inputs,
        grad_outputs=grad_outputs,
        retain_graph=True,
        allow_unused=False,
        create_graph=True
    )
    
    # JVP = Σ(∂f/∂x_i @ v_i) for all inputs i
    jvp_output = None
    for grad, tangent in zip(grads, tangents):
        contribution = grad * tangent
        if jvp_output is None:
            jvp_output = contribution
        else:
            jvp_output = jvp_output + contribution
    
    if y.numel() == 1:
        jvp_output = jvp_output.sum()
    
    return y, jvp_output, {}

def forward_mode_jvp(func, inputs, tangents):
    """
    Forward-mode JVP implementation with forward-mode AD.
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    if not isinstance(tangents, tuple):
        tangents = (tangents,)
    
    if len(inputs) != len(tangents):
        raise ValueError(
            f"Number of inputs ({len(inputs)}) must match number of tangents ({len(tangents)}). "
            "jvp(f, primals, tangents): Expected primals and tangents to have the same python structure. For example, if primals is a tuple of 3 tensors, tangents also must be."
        )
    tangents = list(tangents)

    for i, t in enumerate(tangents):
        if not isinstance(t, (int, float, torch.Tensor)):
            raise RuntimeError(f'jvp(f, primals, tangents): Expected tangents to only contain Tensors, got {type(t)}')
        if not isinstance(t, torch.Tensor):
            new_t = torch.empty_like(inputs[i], device=inputs[i].device, dtype=inputs[i].dtype)
            new_t.fill_(t)
            tangents[i] = new_t
    
    
    with fwAD.dual_level():
        dual_inputs = tuple(
            fwAD.make_dual(inp, tangent) if isinstance(inp, torch.Tensor) else inp
            for inp, tangent in zip(inputs, tangents)
        )
        
        dual_outputs = func(*dual_inputs)
        
        is_outputs_tuple = isinstance(dual_outputs, tuple)
        if not is_outputs_tuple:
            dual_outputs = (dual_outputs,)
        
        outputs = []
        jvp_outputs = []
        
        for dual_out in dual_outputs:
            if isinstance(dual_out, torch.Tensor): # and is_dual_tensor(dual_out):
                primal, tangent = fwAD.unpack_dual(dual_out)
                outputs.append(primal)
                jvp_outputs.append(tangent)
            else:
                # If output is not a dual tensor (e.g., non-tensor output), 
                # use the output as-is and set tangent to zero
                outputs.append(dual_out)
                if isinstance(dual_out, torch.Tensor):
                    jvp_outputs.append(torch.zeros_like(dual_out))
                else:
                    raise ValueError(
                        f"Output {dual_out} is not a tensor and cannot have a tangent component"
                    )
        
        if len(outputs) == 1:
            outputs = outputs[0]
            jvp_output = jvp_outputs[0]
        else:
            outputs = tuple(outputs)
            jvp_output = tuple(jvp_outputs)
    
    return outputs, jvp_output

@contextmanager
def meanflow_jvp_context(parallel_engine: BaseParallelEngine | list[BaseParallelEngine], jvp_mesh_tag: str=None):
    from hy_parallelism.parallel_states import device_mesh_context
    require_unshard_for_jvp = version.parse(torch.__version__) < version.parse('2.8.1')

    if isinstance(parallel_engine, BaseParallelEngine):
        parallel_engine = [parallel_engine]

    with device_mesh_context(jvp_mesh_tag):
        assert not get_parallel_state().pp_enabled, "PP is not supported in jvp"

        if require_unshard_for_jvp:
            for pe in parallel_engine:
                pe.unshard()
        yield
        if require_unshard_for_jvp:
            for pe in parallel_engine:
                pe.reshard()
