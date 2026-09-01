import math
import re

import torch
import scipy.stats as stats


def build_mask_ratio_generator(schedule_type, logger=None):

    if schedule_type in ['linear', 'square', 'cosine', 'arccos']:
        def generator_(shape):
            r = torch.rand(shape)
            if schedule_type == "linear":  # linear scheduler
                ratio = r
            elif schedule_type == "square":  # square scheduler
                ratio = (r ** 2)
            elif schedule_type == "cosine":  # cosine scheduler
                ratio = torch.cos(r * math.pi * 0.5)
            elif schedule_type == "arccos":  # arc cosine scheduler
                ratio = torch.arccos(r) / (math.pi * 0.5)
            else:
                ratio = None
            return ratio

    elif schedule_type.startswith('lognorm'):
        if schedule_type == 'lognorm':
            loc, scale = 0.0, 1.0
        else:
            pattern = re.compile(r"lognorm\(([\d.+-]+),\s*([\d.+]+)\)")
            match = pattern.match(schedule_type)
            assert match, f"Invalid mode: {schedule_type}"
            loc, scale = map(float, match.groups())
        assert scale > 0, f"Invalid lognorm scale: {scale}"

        generator_ = lambda shape: 1 / (1 + torch.exp(-torch.normal(mean=loc, std=scale, size=shape)))

    elif schedule_type.startswith('truncnorm'):
        pattern = re.compile(r"truncnorm\(([\d.+-]+),\s*([\d.+-]+),\s*([\d.+-]+),\s*([\d.+]+)\)")
        match = pattern.match(schedule_type)
        assert match, f"Invalid mode: {schedule_type}"

        a, b, loc, scale = map(float, match.groups())
        assert a < b, f"Invalid truncnorm range: {a} ~ {b}"
        assert scale > 0, f"Invalid truncnorm scale: {scale}"
        if logger is not None:
            logger.info(f"Parse truncnorm parameters: a={a}, b={b}, loc={loc}, scale={scale}")

        truncnorm_gen = stats.truncnorm((a - loc) / scale, (b - loc) / scale, loc=loc, scale=scale)
        generator_ = lambda shape: torch.tensor(truncnorm_gen.rvs(shape), dtype=torch.float32)

    elif schedule_type.startswith('trunc'):
        pattern = re.compile(r"trunc\(([\d.+-]+),\s*([\d.+-]+)\)")
        match = pattern.match(schedule_type)
        assert match, f"Invalid mode: {schedule_type}"

        a, b = map(float, match.groups())
        assert a < b, f"Invalid trunc range: {a} ~ {b}"
        if logger is not None:
            logger.info(f"Parse trunc parameters: a={a}, b={b}")

        generator_ = lambda shape: torch.rand(shape) * (b - a) + a

    else:
        raise ValueError(f"Invalid mode: {schedule_type}")

    if logger is not None:
        logger.info(f"Using mask ratio generator: {schedule_type}")

    return generator_


def adap_sche(step, image_seq_len, mode="arccos", shift=None):
    """ Create a sampling scheduler
       :param
        step  -> int:  number of prediction during inference
        mode  -> str:  the rate of value to unmask
       :return
        scheduler -> torch.LongTensor(): the list of token to predict at each step
    """
    r = torch.linspace(1, 0, step)
    if mode == "root":  # root scheduler
        val_to_mask = 1 - (r ** .5)
    elif mode == "linear":  # linear scheduler
        val_to_mask = 1 - r
    elif mode == "square":  # square scheduler
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":  # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":  # arc cosine scheduler
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    elif mode == "shift":
        val_to_mask = (1 - r) / (1 - (1 - shift) * r)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # fill the scheduler by the ratio of tokens to predict at each step
    sche = (val_to_mask / val_to_mask.sum()) * image_seq_len
    sche = sche.round()
    sche[sche == 0] = 1  # add 1 to predict a least 1 token / step
    sche[-1] += image_seq_len - sche.sum()  # need to sum up nb of code
    return sche.int()


def get_mask_code(generator, code, mask_id, user_mask=None, use_generator=True):
    """ Replace the code token by *value* according the ratio generator

    Parameters
    ----------
    generator: callable
        a function that generate the mask ratio
    code: torch.Tensor
        [bs, ...], the unmasked code
    mask_id: int
        mask the code by the value
    user_mask: torch.Tensor
        User predefined mask, will be merged with generated mask.

    Returns
    -------
    masked_code: torch.Tensor
        [bs, ...], the masked version of the code
    mask: torch.Tensor
        [bs, ...], the binary mask of the mask
    """
    if use_generator:
        mask_ratio = generator(shape=(code.size(0),))
    else:
        mask_ratio = torch.zeros((code.size(0),))

    mask_code = code.detach().clone()
    # Sample the amount of tokens + localization to mask
    ratio_shape = [code.size(0)] + [1] * (len(code.size()) - 1)
    mask = torch.rand(size=code.size()) < mask_ratio.view(*ratio_shape)
    if user_mask is not None:
        mask = mask | user_mask.bool()

    mask_code[mask] = torch.full_like(mask_code[mask], mask_id)

    return mask_code, mask


def inverse_mask(mask, dtype=torch.float32):
    inverted_mask = 1.0 - mask.type(dtype)
    inverted_mask = inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )
    return inverted_mask


def create_attention_mask_t2i(
        sequence, pad_id, boi_id, eoi_id, mask_pad=False, return_inverse_mask=True, dtype=torch.float32,
):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # Masks to identify different types of tokens
    is_start_image = sequence == boi_id
    is_end_image = sequence == eoi_id

    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image

    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)
    mask = in_image_segment[:, :, None] | causal_mask[None, :, :]

    if mask_pad:
        for i in range(mask.shape[0]):
            pad_end_idx = torch.where(sequence[i] == pad_id)[0]
            if len(pad_end_idx) != 0:
                idx = pad_end_idx[-1] + 1
                mask[i][idx:, :idx] = 0    # Fill left bottom corner

    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        mask = inverse_mask(mask, dtype)
    return mask.unsqueeze(1)


def create_attention_mask_general(
        sequence, pad_id, slice_s_image_ranges, mask_pad=False, return_inverse_mask=True, dtype=torch.float32,
        pad_endpoint=None,
):
    """
    Examples
    --------
    ```python
    txlen = 10
    imlen = 10
    sequence = torch.zeros((1, txlen+imlen+imlen))
    sequence[0, :3] = 4
    sequence[0, txlen-5] = 1
    sequence[0, txlen-5 + imlen + 1] = 2
    sequence[0, txlen-5 + imlen + 2] = 1
    sequence[0, txlen-5 + imlen + 2 + imlen + 1] = 2
    sequence[0, txlen-4:txlen-4+imlen] = 3
    sequence[0, txlen-4+imlen+2:txlen-4+imlen+2+imlen] = 3

    mask = create_attention_mask_general(
        sequence, 4, [slice(txlen-4, txlen-4+imlen), slice(txlen-4+imlen+2, txlen-4+imlen+2+imlen)],
        mask_pad=True
    )
    ```
    """
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape
    mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)
    for s_image_range in slice_s_image_ranges:
        mask[s_image_range, s_image_range] = 1
    mask = mask[None]

    if mask_pad:
        if pad_endpoint is None:
            pad_endpoint = L
        for i in range(mask.shape[0]):
            pad_end_idx = torch.where(sequence[i, :pad_endpoint] == pad_id)[0]
            if len(pad_end_idx) != 0:
                idx = pad_end_idx[-1] + 1
                mask[i][idx:, :idx] = 0    # Fill left bottom corner

    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        mask = inverse_mask(mask, dtype)
    return mask.unsqueeze(1)


def create_attention_mask_mmu(
        sequence, eoi_pos, return_inverse_mask=True
):
    N, L = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(sequence.device)
    causal_mask[:, :, :, :eoi_pos + 1] = 1

    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        mask_text = inverse_mask(causal_mask, sequence.dtype)
    return causal_mask.unsqueeze(1)
