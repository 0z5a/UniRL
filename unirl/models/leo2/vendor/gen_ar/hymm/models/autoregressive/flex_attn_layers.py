import torch


def causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def slices_to_ids_tensor(image_slices, n_tokens, device):
    image_mask = torch.ones((n_tokens,), dtype=torch.int16, device=device) * (-1)
    for i_, sli_ in enumerate(image_slices):
        image_mask[sli_] = i_
    return image_mask


def offsets_to_ids_tensor(offsets, n_tokens, device):
    repeats = torch.diff(offsets)
    indices = torch.arange(len(offsets) - 1)
    expanded = torch.repeat_interleave(indices, repeats)
    # use int16 to avoid overflow when too many images
    mask = torch.zeros((n_tokens,), dtype=torch.int16, device=device) - 1
    mask[:len(expanded)] = expanded
    return mask


def create_causal_mask_mod(n_tokens, device, offsets=None):
    if offsets is None:
        return causal_mask_mod

    else:
        sample_id = offsets_to_ids_tensor(offsets, n_tokens, device)

        def causal_mask_mod_with_sample(b, h, q_idx, kv_idx):
            same_sample = sample_id[q_idx] == sample_id[kv_idx]
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return inner_mask & same_sample

        return causal_mask_mod_with_sample


def create_batch_causal_mask_mod(n_tokens, device, batch_offsets=None):
    if batch_offsets is None:
        return causal_mask_mod

    else:
        sample_id = torch.stack([
            offsets_to_ids_tensor(offsets, n_tokens, device)
            for offsets in batch_offsets
        ])

        def causal_mask_mod_with_sample(b, h, q_idx, kv_idx):
            same_sample = sample_id[b][q_idx] == sample_id[b][kv_idx]
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return inner_mask & same_sample

        return causal_mask_mod_with_sample


def create_interleave_mask_mod(gen_image_slices, src_image_slices, hole_slices, n_tokens, device, offsets=None):
    image_id = slices_to_ids_tensor(gen_image_slices + src_image_slices, n_tokens, device)
    hole_id = slices_to_ids_tensor(hole_slices, n_tokens, device)

    if offsets is None:
        def interleave_mask_mod(b, h, q_idx, kv_idx):
            same_image = (image_id[q_idx] == image_id[kv_idx]) & (image_id[q_idx] >= 0) & (image_id[kv_idx] >= 0)
            holes = (hole_id[q_idx] != hole_id[kv_idx]) & (hole_id[kv_idx] >= 0) & (q_idx > kv_idx)
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return (same_image | inner_mask) & (~holes)

    else:
        sample_id = offsets_to_ids_tensor(offsets, n_tokens, device)

        def interleave_mask_mod(b, h, q_idx, kv_idx):
            same_sample = sample_id[q_idx] == sample_id[kv_idx]
            same_image = (image_id[q_idx] == image_id[kv_idx]) & (image_id[q_idx] >= 0) & (image_id[kv_idx] >= 0)
            holes = (hole_id[q_idx] != hole_id[kv_idx]) & (hole_id[kv_idx] >= 0) & (q_idx > kv_idx)
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return (same_image | inner_mask) & (~holes) & same_sample

    return interleave_mask_mod


def create_batch_interleave_mask_mod(
        batch_gen_image_slices, batch_src_image_slices, batch_hole_slices, n_tokens, device,
        batch_offsets=None,
):
    image_id = torch.stack([
        slices_to_ids_tensor(gen_image_slices + src_image_slices, n_tokens, device)
        for gen_image_slices, src_image_slices in zip(batch_gen_image_slices, batch_src_image_slices)
    ])
    hole_id = torch.stack([
        slices_to_ids_tensor(hole_slices, n_tokens, device)
        for hole_slices in batch_hole_slices
    ])

    if batch_offsets is None:

        def interleave_mask_mod(b, h, q_idx, kv_idx):
            same_image = (image_id[b][q_idx] == image_id[b][kv_idx]) & (image_id[b][q_idx] >= 0) & (image_id[b][kv_idx] >= 0)
            holes = (hole_id[b][q_idx] != hole_id[b][kv_idx]) & (hole_id[b][kv_idx] >= 0) & (q_idx > kv_idx)
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return (same_image | inner_mask) & (~holes)

    else:
        sample_id = torch.stack([
            offsets_to_ids_tensor(offsets, n_tokens, device)
            for offsets in batch_offsets
        ])

        def interleave_mask_mod(b, h, q_idx, kv_idx):
            same_sample = sample_id[b][q_idx] == sample_id[b][kv_idx]
            same_image = (image_id[b][q_idx] == image_id[b][kv_idx]) & (image_id[b][q_idx] >= 0) & (image_id[b][kv_idx] >= 0)
            holes = (hole_id[b][q_idx] != hole_id[b][kv_idx]) & (hole_id[b][kv_idx] >= 0) & (q_idx > kv_idx)
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return (same_image | inner_mask) & (~holes) & same_sample

    return interleave_mask_mod


def create_text_image_mask_mod(image_slices, n_tokens, device, offsets=None):
    image_id = slices_to_ids_tensor(image_slices, n_tokens, device)

    if offsets is None:
        def text_image_mask_mod(b, h, q_idx, kv_idx):
            same_image = (image_id[q_idx] == image_id[kv_idx]) & (image_id[q_idx] >= 0) & (image_id[kv_idx] >= 0)
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return same_image | inner_mask

    else:
        sample_id = offsets_to_ids_tensor(offsets, n_tokens, device)

        def text_image_mask_mod(b, h, q_idx, kv_idx):                
            same_sample = sample_id[q_idx] == sample_id[kv_idx]
            same_image = (image_id[q_idx] == image_id[kv_idx]) & (image_id[q_idx] >= 0) & (image_id[kv_idx] >= 0)
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return (same_image | inner_mask) & same_sample

    return text_image_mask_mod


def create_batch_text_image_mask_mod(batch_image_slices, n_tokens, device, batch_offsets=None):
    image_id = torch.stack([
        slices_to_ids_tensor(image_slices, n_tokens, device)
        for image_slices in batch_image_slices
    ])

    if batch_offsets is None:
        def text_image_mask_mod(b, h, q_idx, kv_idx):
            same_image = (image_id[b][q_idx] == image_id[b][kv_idx]) & (image_id[b][q_idx] >= 0) & (image_id[b][kv_idx] >= 0)
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return same_image | inner_mask

    else:
        sample_id = torch.stack([
            offsets_to_ids_tensor(offsets, n_tokens, device)
            for offsets in batch_offsets
        ])

        def text_image_mask_mod(b, h, q_idx, kv_idx):
            same_sample = sample_id[b][q_idx] == sample_id[b][kv_idx]
            same_image = (image_id[b][q_idx] == image_id[b][kv_idx]) & (image_id[b][q_idx] >= 0) & (
                        image_id[b][kv_idx] >= 0)
            inner_mask = causal_mask_mod(b, h, q_idx, kv_idx)
            return (same_image | inner_mask) & same_sample

    return text_image_mask_mod


if __name__ == "__main__":
    print(offsets_to_ids_tensor(
        torch.tensor([0, 4, 10]),
        n_tokens=12,
        device=torch.device('cpu')
    ))
