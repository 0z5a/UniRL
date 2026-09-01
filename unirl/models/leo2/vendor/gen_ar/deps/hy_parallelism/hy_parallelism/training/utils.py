import torch

def attatch_loss_to_activation(activation, loss):
    """
    A hack to attach loss to activation, allowing to add extra loss without modifying the loss function.
    This is useful when the loss function is complex or computed in hierarchical manner (like MOE aux loss).

    Explanation:
        Given a loss function `F`, and an activation,

        F( attatch_loss_to_activation(activation, loss) ).backward()
        Equals to: ( F(activation) + loss.sum() ).backward()

        `loss` can be a tensor of arbitrary shape; its shape does not need to match `activation`
    """

    class LossAttachment(torch.autograd.Function):

        @staticmethod
        def forward(ctx, output: torch.Tensor, loss: torch.Tensor):
            ctx.save_for_backward(loss)
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            (loss,) = ctx.saved_tensors
            return grad_output, torch.ones_like(loss)

    return LossAttachment.apply(activation, loss)
