

import loguru
import torch
from torchtitan.models.moe import MoE
from hy_parallelism.training.jvp_utils import in_jvp_context, tolist, torch_jvp_tolist_bug_is_fixed
from hy_parallelism.training.jvp_utils import safe_jvp_op

def patch_moe_reset_parameters():

    def reset_parameters(self, init_std=0.02):
        self.init_weights(init_std=0.02, buffer_device=torch.device('cuda'))

    MoE.reset_parameters = reset_parameters
    MoE.init_weights
    

def patch_for_jvp():
    # remove inplace operation in moe.forward to support jvp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # top_scores and selected_experts_indices shape (bs*slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            if not in_jvp_context():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # top_scores_experts_sorted and token_indices_experts_sorted shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted // self.router.top_k]

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # shared expert
        # Note: we execute the shared expert before scoring the output of the routed expert
        # to "implicitly" overlap the shared expert compute with token combine communication
        out = self.shared_experts(x) if self.shared_experts is not None else None

        # Unsort routed outputs
        routed_output_unsorted = torch.zeros(
            (bs * slen * self.router.top_k, dim),
            dtype=routed_output.dtype,
            device=routed_output.device,
        )
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.reshape(
            -1, self.router.top_k, dim
        )
        if not self.score_before_experts:
            out_experts = (
                torch.bmm(
                    top_scores.reshape(-1, 1, self.router.top_k),
                    routed_output_unsorted.float(),
                )
                .to(x.dtype)
                .squeeze(1)
            )
        else:
            out_experts = routed_output_unsorted.sum(dim=1)

        if out is None:
            return out_experts.reshape(bs, slen, dim)
        return (out + out_experts).reshape(bs, slen, dim)

    from torchtitan.models.moe import moe

    # NOTE: keeping this for-loop implementation for comparison
    #       and readability, may remove later
    def _run_experts_for_loop(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: this would incur a synchronization between device and host
        if in_jvp_context():
            num_tokens_per_expert_list = tolist(num_tokens_per_expert)
        else:
            num_tokens_per_expert_list = num_tokens_per_expert.tolist()

        # side-effect code due to the usage of generate_permute_indices
        num_padding = x.shape[0] - sum(num_tokens_per_expert_list)

        # a tuple of tensors indexed by experts
        # each with shape (tokens_per_expert(varying), dim)
        x_splits = torch.split(
            x[: sum(num_tokens_per_expert_list)],
            split_size_or_sections=num_tokens_per_expert_list,
            dim=0,
        )
        out_experts_splits = []
        for expert_idx, x_expert in enumerate(x_splits):
            h = moe.F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
            h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
            h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
            # h shape (tokens_per_expert(varying), dim)
            out_experts_splits.append(h)
        out = torch.cat(out_experts_splits, dim=0)

        # side-effect code due to the usage of generate_permute_indices
        out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

        return out

    # safe_wait_tensor = safe_jvp_op(torch.ops._c10d_functional.wait_tensor)

    # safe_wait_tensor.__dict__['default'] = torch.ops._c10d_functional.wait_tensor.default
    # safe_wait_tensor.__dict__['__qualname__'] = torch.ops._c10d_functional.wait_tensor.__qualname__
    # safe_wait_tensor.__dict__['default'] = safe_wait_tensor
    # safe_wait_tensor.__dict__['__qualname__'] = 'my_wait_tensor'
    # torch.ops._c10d_functional.wait_tensor = safe_wait_tensor

    # from hy_parallelism.training.jvp_utils import safe_jvp_op2
    # from torch.distributed import _functional_collectives
    # _functional_collectives.wait_tensor = safe_jvp_op2(_functional_collectives.wait_tensor)
    # torch.ops._c10d_functional.wait_tensor = safe_jvp_op2(torch.ops._c10d_functional.wait_tensor)

    # from torch.distributed import _functional_collectives

    # 不行
    # _functional_collectives.wait_tensor = safe_jvp_op2(_functional_collectives.wait_tensor)
    # 可以
    # _functional_collectives._maybe_wrap_tensor = safe_jvp_op2(_functional_collectives._maybe_wrap_tensor)

    # torch.Tensor.tolist = safe_jvp_op(torch.Tensor.tolist)

    """
    MoE.forward = forward # add_
    moe._run_experts_for_loop = _run_experts_for_loop
    """


    # assert torch_jvp_tolist_bug_is_fixed()
    # loguru.logger.info('patch_for_jvp done')
    # exit()


    from typing import Optional
    from torch._C._distributed_c10d import _resolve_process_group

    from torch import distributed as dist
    class AllToAllGivenGroupName(torch.autograd.Function):
        @staticmethod
        def forward(input, output_split_sizes, input_split_sizes, group_name):
            group = _resolve_process_group(group_name)

            world_size = dist.get_world_size(group=group)

            if world_size == 1:
                return input

            input = input.contiguous()

            if output_split_sizes is None:
                output = torch.empty_like(input)
            else:
                output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
            dist.all_to_all_single(
                output,
                input,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            )
            return output

        @staticmethod
        def backward(ctx, *grad_output):
            return (
                AllToAllGivenGroupName.apply(*grad_output, ctx.input_split_sizes, ctx.output_split_sizes, ctx.group_name),
                None,
                None,
                None,
            )
        
        @staticmethod
        def setup_context(ctx, inputs, output):
            input, output_split_sizes, input_split_sizes, group_name = inputs
            ctx.group_name = group_name
            ctx.output_split_sizes = output_split_sizes
            ctx.input_split_sizes = input_split_sizes

        @staticmethod
        def jvp(ctx, input_tangent, output_split_sizes_tangent, input_split_sizes_tangent, group_name_tangent):
            # Only the input tensor has a meaningful tangent
            if input_tangent is None:
                return None
                
            # Apply the same AllToAll transformation to the tangent vector
            return AllToAllGivenGroupName.apply(input_tangent, ctx.output_split_sizes, ctx.input_split_sizes, ctx.group_name)

    from torch.distributed._functional_collectives import _maybe_wrap_tensor
    from torch.distributed import _functional_collectives
    class _FromTorchTensor(_functional_collectives._FromTorchTensor):
        
        @staticmethod
        def jvp(ctx, input_tangent):
            return _FromTorchTensor.apply(input_tangent)


    # HACK(kevinkhwu): Fix bug
    # /opt/python3.12/lib/python3.12/site-packages/torch/autograd/graph.py:841: UserWarning: _c10d_functional::all_to_all_single: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at /root/source/pytorch/torch/csrc/autograd/autograd_not_implemented_fallback.cpp:62.)
    torch.ops._c10d_functional_autograd.all_to_all_single = AllToAllGivenGroupName.apply
    _functional_collectives._FromTorchTensor = _FromTorchTensor


