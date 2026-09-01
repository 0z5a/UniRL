from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .torch_utils import recursively_apply


def prepare_generator(
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]],
        target_device: Optional["torch.device"] = None,
):
    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if target_device is not None and gen_device_type != target_device.type:
            raise ValueError(f"Cannot generate a {target_device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    return generator


def uniform_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
):
    """A helper function to create random uniform tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    batch_size = shape[0]
    generator = prepare_generator(generator, device)

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.rand(*shape, generator=generator[i], device=device or generator[i].device, dtype=dtype)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0)
    else:
        latents = torch.rand(*shape, generator=generator, device=device or generator.device, dtype=dtype)

    return latents


def categorical_sample(
        probs: torch.Tensor,
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
):
    """
    Reimplementation of `torch.distributions.Categorical.sample` to allow for generator to be passed.
    """
    probs = probs / probs.sum(-1, keepdim=True)
    batch_shape, event_shape = probs.size()[:-1], probs.size()[-1]

    generator = prepare_generator(generator, probs.device)
    if isinstance(generator, list):
        sample = torch.stack(
            [
                torch.multinomial(probs[i], 1, replacement=True, generator=generator[i])
                for i in range(probs.size(0))
            ]
        )
    else:
        sample = torch.multinomial(probs.view(-1, event_shape), 1, replacement=True, generator=generator)
    sample = sample.view(*batch_shape)

    return sample


def gumbel_sample(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
):
    """
    Reimplementation of `torch.distributions.Gumbel.sample` to allow for generator to be passed.
    """
    uniform = uniform_tensor(shape, generator=generator, device=device, dtype=dtype)
    gumbel = -torch.log(-torch.log(uniform + 1e-16) + 1e-16)
    return gumbel


# Modified from: https://github.com/huggingface/accelerate/blob/74e08e5205501826c0d4bae5abb7c25cb3da5e15/src/accelerate/utils/operations.py#L320
def gather_tensor(tensor):
    """
    Recursively gather tensor in a nested list/tuple/dictionary of tensors from all devices.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to gather.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """
    is_distributed = dist.is_initialized()
    backend = dist.get_backend() if is_distributed else None
    world_size = dist.get_world_size() if is_distributed else 1
    device = tensor.device

    def _gpu_gather_one(tensor):
        nonlocal backend, world_size, device

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if not is_distributed:
            return [tensor]

        if backend != "gloo":
            # We use `empty` as `all_gather_into_tensor` slightly
            # differs from `all_gather` for better efficiency,
            # and we rely on the number of items in the tensor
            # rather than its direct shape
            output_shape = (world_size * tensor.numel(),) + tensor.shape[1:]
            output_tensor = torch.empty(
                output_shape,
                dtype=tensor.dtype,
                device=device
            )
            dist.all_gather_into_tensor(output_tensor, tensor)

            return output_tensor.view(world_size, *tensor.shape)
        else:
            # a backend of `None` is always CPU
            # also gloo does not support `all_gather_into_tensor`,
            # which will result in a larger memory overhead for the op
            gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(gather_list, tensor)
            return torch.cat(gather_list, dim=0)

    try:
        return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)
    except Exception as e:
        raise RuntimeError(f"Error gathering tensor(s): {str(e)}") from e
