import numpy as np
import torch
from typing import Union, Tuple, List


def _to_tuple(x, dim=2):
    if isinstance(x, (int, np.integer)):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2, with_end=False):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.
        with_end (bool): If True, include the stop value in the meshgrid. Defaults to False.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
        # assert num are all integers
        num_int = [int(x) for x in num]
        assert (torch.tensor(num) == torch.tensor(num_int)).all(), f"num should be int, but got {num}"
        num = num_int
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)       # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)      # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)       # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        if with_end:
            g = torch.linspace(a, b, n, dtype=torch.float32)
        else:
            g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")   # dim x [H, W]
    grid = torch.stack(grid, dim=0)     # [dim, H, W]

    return grid


def get_1d_rotary_pos_embed(dim: int,
                            pos: Union[torch.FloatTensor, int],
                            theta: float = 10000.0,
                            use_real: bool = True,
                            theta_rescale_factor: float = 1.0,
                            interpolation_factor: float = 1.0,
                            interleave: bool = False,
                            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.
        interpolation_factor (float, optional): Interpolation factor for frequency tensor. Defaults to 1.0.
        interleave (bool): If True, interleave the real and imaginary parts. Defaults to False.

    Returns:
        freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
    # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
    freqs = torch.outer(pos * interpolation_factor, freqs)                          # [S, D/2]
    if interleave:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)     # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)     # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_repeat = freqs.repeat(1, 2)
        freqs_cos = freqs_repeat.cos()
        freqs_sin = freqs_repeat.sin()
        return freqs_cos, freqs_sin


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L80

def get_nd_rotary_pos_embed(rope_dim_list, start, *args, theta=10000., use_real=True,
                            theta_rescale_factor: Union[float, List[float]] = 1.0,
                            interpolation_factor: Union[float, List[float]] = 1.0,
                            interleave: bool = False,
                            return_concat: bool = True,
                            ):
    """
    This is RoPE generation API for tokens with n-d structure. N-d positional embeddings are distributed with
    shared frequencies.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.
        interpolation_factor (float): Interpolation factor for frequency tensor. Defaults to 1.0.
        interleave (bool): If True, interleave the real and imaginary parts. Defaults to False.
        return_concat (bool): If True, return concatenated nd embeddings. Otherwise, return a list of nd embeddings.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """

    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))   # [3, D, H, W] / [2, H, W]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(rope_dim_list), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(rope_dim_list), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(rope_dim_list[i], grid[i].reshape(-1), theta,
                                      theta_rescale_factor=theta_rescale_factor[i],
                                      interpolation_factor=interpolation_factor[i],
                                      interleave=interleave,
                                      )    # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    if return_concat:
        cos = torch.cat([emb[0] for emb in embs], dim=1)    # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)    # (WHD, D/2)
        return cos, sin
    else:
        cos = [emb[0] for emb in embs]
        sin = [emb[1] for emb in embs]
        return cos, sin


def get_mlm_rope(rope_dim_list, height, width, max_len, img_pos, device, shift_image=False, add_batch_axis=False, **kwargs):
    """ 3d rope implementation. Designed by jarvizhang.
    Text position encoded in the first d1 channels, image positions encoded in the last d2 * 2 channels.
    And we let head_size = d1 + d2 * 2.

    max_len: Treat each image as a single token, then max_len is the total number of tokens.
    img_pos: The absolute position of the image in the sequence with `max_len` tokens.

    If there are multiple images, the `height`, `width`, `img_pos` should be list of int with each element
    corresponding to the height, width and position of each image.
    """
    if isinstance(height, list):
        assert len(height) == len(width), f"height and width should be equal length, got {height} and {width}"
        assert len(height) == len(img_pos), f"height and img_pos should be equal length, got {height} and {img_pos}"
    else:
        height = [height]
        width = [width]
        img_pos = [img_pos]

    text_rope_sizes = [max_len, 1, 1]
    text_cos, text_sin = get_nd_rotary_pos_embed(rope_dim_list, text_rope_sizes, **kwargs)

    cos_list, sin_list = [], []
    cur_h, cur_w, cur_p = 0, 0, 0
    for h, w, abs_p in zip(height, width, img_pos):
        image_rope_starts = [abs_p, cur_h, cur_w]
        image_rope_ends = [abs_p + 1, cur_h + h, cur_w + w]
        image_cos, image_sin = get_nd_rotary_pos_embed(rope_dim_list, image_rope_starts, image_rope_ends, **kwargs)
        cos_list.extend([text_cos[cur_p:abs_p], image_cos])
        sin_list.extend([text_sin[cur_p:abs_p], image_sin])
        if shift_image:
            cur_h, cur_w, cur_p = h + 1, w + 1, abs_p + 1
        else:
            cur_h, cur_w, cur_p = 0, 0, abs_p + 1

    freqs_cos = torch.cat(cos_list + [text_cos[cur_p:]]).to(device)
    freqs_sin = torch.cat(sin_list + [text_sin[cur_p:]]).to(device)

    if add_batch_axis:
        freqs_cos = freqs_cos.unsqueeze(0)
        freqs_sin = freqs_sin.unsqueeze(0)

    return freqs_cos, freqs_sin


# Alias for general usage by mar and transfusion
get_3d_rope = get_mlm_rope


def get_exclusive_nd_rotary_pos_embed(dim, start, *args, theta=10000., use_real=True,
                                      theta_rescale_factor: float = 1.0,
                                      interpolation_factor: float = 1.0,
                                      interleave: bool = False,
                                      return_concat: bool = True,
                                      ):
    """
    This is RoPE generation API for tokens with n-d structure. N-d positional embeddings are distributed in
    exclusive frequencies.

    Args:
        dim (int): Dimension of rope.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.
        interpolation_factor (float): Interpolation factor for frequency tensor. Defaults to 1.0.
        interleave (bool): If True, interleave the real and imaginary parts. Defaults to False.
        return_concat (bool): If True, return concatenated nd embeddings. Otherwise, return a list of nd embeddings.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """
    num_axes = 1 if isinstance(start, int) else len(start)
    assert dim % (num_axes * 2) == 0, (
        f"dim ({dim}) should be divisible by num_axes ({num_axes}) * 2, but got dim % (num_axes * 2) = {dim % (num_axes * 2)}"
    )

    grid = get_meshgrid_nd(start, *args, dim=num_axes)   # [2, H, W]
    points = grid.reshape(num_axes, -1).transpose(0, 1).unsqueeze(1)  # [HW, 1, 2]

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = (points * interpolation_factor) * freqs.view(1, dim // 2 // num_axes, num_axes)  # [HW, D/2/num_axes, num_axes]
    freqs = freqs.reshape(-1, dim // 2)  # [HW, D/2]
    if interleave:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_repeat = freqs.repeat(1, 2)
        freqs_cos = freqs_repeat.cos()
        freqs_sin = freqs_repeat.sin()
        return freqs_cos, freqs_sin


def get_mlm_rope_exclusive(dim, height, width, max_len, img_pos, device=None, **kwargs):
    """ Shared 2d rope implementation. Designed by JianLinSu.
    Text and image share the same channels. Text position expanded from n to (n, n), and keep the theta_2j and
    theta_2j+1 different. Image position use the same theta as text.

    References
    ----------
    https://kexue.fm/archives/10352
    """
    cos, sin = get_1d_rotary_pos_embed(dim, max_len, **kwargs)

    image_rope_starts = [img_pos + 0.5 * (height * width - height), img_pos + 0.5 * (height * width - width)]
    image_rope_ends = [img_pos + 0.5 * (height * width + height), img_pos + 0.5 * (height * width + width)]

    image_cos, image_sin = get_exclusive_nd_rotary_pos_embed(
        dim, image_rope_starts, image_rope_ends, [height, width], **kwargs)

    cos[img_pos:img_pos + height * width] = image_cos
    sin[img_pos:img_pos + height * width] = image_sin

    if device is not None:
        cos, sin = cos.to(device), sin.to(device)

    return cos, sin
