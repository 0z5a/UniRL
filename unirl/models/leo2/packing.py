"""Pack independent Leo2 full-sequence condition blobs for one FA3 forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

_FULL_SEQUENCE_MASKS = (
    "visual_mask",
    "text_mask",
    "audio_mask",
    "cond_text_scatter_mask",
)
_TOKEN_INDEX_FIELDS = (
    "und_token_indices",
    "gen_token_indices",
    "audio_token_indices",
)
_UNSUPPORTED_PACKED_FIELDS = (
    "cond_vae_images",
    "cond_vae_mask",
    "cond_timesteps",
    "timesteps_index",
)


@dataclass(frozen=True)
class PackedLeo2Conditions:
    """The static inputs shared by every denoising step of a packed batch."""

    input_ids: torch.Tensor
    model_kwargs: dict[str, Any]
    packing_kwargs: dict[str, Any]
    batch_size: int
    sequence_lengths: tuple[int, ...]


def _expect_row(tensor: Any, *, name: str, length: int | None = None) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Leo2 packed forward expected Tensor {name}, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError(f"Leo2 packed forward expected {name} shaped [1,S], got {tuple(tensor.shape)}")
    if length is not None and tensor.shape[1] != length:
        raise ValueError(f"Leo2 packed forward expected {name} sequence length {length}, got {tensor.shape[1]}")
    return tensor


def _same_tensor_layout(tensors: Sequence[torch.Tensor], *, name: str) -> None:
    first = tensors[0]
    for index, tensor in enumerate(tensors[1:], start=1):
        if tensor.device != first.device or tensor.dtype != first.dtype:
            raise ValueError(
                f"Leo2 packed forward requires matching {name} device/dtype; "
                f"sample 0={first.device}/{first.dtype}, sample {index}={tensor.device}/{tensor.dtype}"
            )


def _pack_optional_rows(
    kwargs: Sequence[dict[str, Any]],
    *,
    key: str,
    sequence_lengths: Sequence[int],
) -> torch.Tensor | None:
    values = [item.get(key) for item in kwargs]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Leo2 packed forward requires {key} in either every sample or no sample")
    rows = [
        _expect_row(value, name=f"model_kwargs.{key}[{index}]", length=sequence_lengths[index])
        for index, value in enumerate(values)
    ]
    _same_tensor_layout(rows, name=key)
    return torch.cat(rows, dim=1)


def _pack_token_indices(
    kwargs: Sequence[dict[str, Any]],
    *,
    key: str,
    sequence_lengths: Sequence[int],
    offsets: Sequence[int],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    values = [item.get(key) for item in kwargs]
    if all(value is None for value in values):
        return None, None
    if any(value is None for value in values):
        raise ValueError(f"Leo2 packed forward requires {key} in either every sample or no sample")

    rows: list[torch.Tensor] = []
    lengths: list[int] = []
    for index, value in enumerate(values):
        row = _expect_row(value, name=f"model_kwargs.{key}[{index}]")
        if row.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"Leo2 packed forward expected integer {key}, got sample {index} dtype={row.dtype}")
        if row.numel():
            minimum, maximum = torch.aminmax(row)
            if int(minimum) < 0 or int(maximum) >= sequence_lengths[index]:
                raise ValueError(f"Leo2 packed forward {key}[{index}] falls outside [0,{sequence_lengths[index]})")
        # Normalize valid int32 cache indices for PyTorch gather/scatter.
        rows.append(row.to(dtype=torch.long) + offsets[index])
        lengths.append(row.shape[1])

    _same_tensor_layout(rows, name=key)
    packed = torch.cat(rows, dim=1)
    token_lengths = torch.tensor(lengths, dtype=torch.long, device=packed.device)
    return packed, token_lengths


def _pack_text_conditions(kwargs: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    states = [item.get("cond_text_states") for item in kwargs]
    masks = [item.get("cond_text_mask") for item in kwargs]
    if any(not isinstance(value, torch.Tensor) for value in states):
        raise TypeError("Leo2 packed forward requires Tensor cond_text_states in every sample")
    if any(not isinstance(value, torch.Tensor) for value in masks):
        raise TypeError("Leo2 packed forward requires Tensor cond_text_mask in every sample")

    state_tensors: list[torch.Tensor] = []
    mask_tensors: list[torch.Tensor] = []
    hidden_size: int | None = None
    for index, (state, mask) in enumerate(zip(states, masks)):
        if state.ndim != 3 or state.shape[0] != 1:
            raise ValueError(
                "Leo2 packed forward expected cond_text_states shaped [1,T,H], "
                f"got sample {index} shape={tuple(state.shape)}"
            )
        mask = _expect_row(mask, name=f"model_kwargs.cond_text_mask[{index}]", length=state.shape[1])
        if hidden_size is None:
            hidden_size = state.shape[2]
        elif state.shape[2] != hidden_size:
            raise ValueError(
                f"Leo2 packed forward text hidden sizes differ: sample 0={hidden_size}, sample {index}={state.shape[2]}"
            )

        # The eval full-sequence path requires explicitly compacted text states.
        valid = mask[0].bool()
        state_tensors.append(state[:, valid, :])
        mask_tensors.append(torch.ones((1, int(valid.sum())), dtype=mask.dtype, device=mask.device))

    _same_tensor_layout(state_tensors, name="cond_text_states")
    _same_tensor_layout(mask_tensors, name="cond_text_mask")
    return torch.cat(state_tensors, dim=1), torch.cat(mask_tensors, dim=1)


def _pack_rope_media_info(
    kwargs: Sequence[dict[str, Any]],
    *,
    sequence_lengths: Sequence[int],
    offsets: Sequence[int],
) -> list[list[tuple[slice, Any, Any]]]:
    packed: list[tuple[slice, Any, Any]] = []
    for sample_index, item in enumerate(kwargs):
        infos = item.get("rope_media_info")
        if not isinstance(infos, list) or len(infos) != 1 or not isinstance(infos[0], list):
            raise ValueError(
                "Leo2 packed forward expects each rope_media_info to contain one physical row; "
                f"sample {sample_index} got {type(infos).__name__} with length "
                f"{len(infos) if isinstance(infos, list) else 'n/a'}"
            )
        for info_index, info in enumerate(infos[0]):
            if not isinstance(info, tuple) or len(info) != 3 or not isinstance(info[0], slice):
                raise TypeError(f"Leo2 packed forward invalid rope_media_info[{sample_index}][{info_index}]")
            media_slice, token_shape, metadata = info
            if (
                type(media_slice.start) is not int
                or type(media_slice.stop) is not int
                or media_slice.start < 0
                or media_slice.stop < media_slice.start
                or media_slice.stop > sequence_lengths[sample_index]
            ):
                raise ValueError(
                    f"Leo2 packed forward rope slice {media_slice!r} is outside "
                    f"sample {sample_index} length {sequence_lengths[sample_index]}"
                )
            packed.append(
                (
                    slice(
                        media_slice.start + offsets[sample_index],
                        media_slice.stop + offsets[sample_index],
                        media_slice.step,
                    ),
                    token_shape,
                    metadata,
                )
            )
    return [packed]


def pack_hymm_conditions(
    blobs: Sequence[dict[str, Any]],
    *,
    sequence_parallel_size: int = 1,
) -> PackedLeo2Conditions:
    """Flatten B native single-sample blobs into one isolated packed sequence."""

    if type(sequence_parallel_size) is not int or sequence_parallel_size < 1:
        raise ValueError(f"Leo2 condition packing requires sequence_parallel_size >= 1, got {sequence_parallel_size!r}")
    if not isinstance(blobs, (list, tuple)) or len(blobs) < 1:
        raise ValueError("Leo2 condition packing requires at least one sample blob")
    if any(not isinstance(blob, dict) for blob in blobs):
        raise TypeError("Leo2 condition packing requires a sequence of dict blobs")

    input_ids = [_expect_row(blob.get("input_ids"), name=f"input_ids[{index}]") for index, blob in enumerate(blobs)]
    _same_tensor_layout(input_ids, name="input_ids")
    sequence_lengths = tuple(row.shape[1] for row in input_ids)
    offsets = [0]
    for length in sequence_lengths:
        if length <= 0:
            raise ValueError("Leo2 packed forward does not support empty sample sequences")
        offsets.append(offsets[-1] + length)

    kwargs = [blob.get("model_kwargs") for blob in blobs]
    if any(not isinstance(item, dict) for item in kwargs):
        raise TypeError("Leo2 condition packing requires model_kwargs dict in every sample")
    for sample_index, item in enumerate(kwargs):
        unsupported = [key for key in _UNSUPPORTED_PACKED_FIELDS if item.get(key) is not None]
        if unsupported:
            raise ValueError(
                "Leo2 packed forward currently supports the T2V/AV full-sequence path; "
                f"sample {sample_index} has unsupported non-null fields {unsupported}"
            )
        attention = _expect_row(
            item.get("attention_mask"),
            name=f"model_kwargs.attention_mask[{sample_index}]",
            length=sequence_lengths[sample_index],
        )
        if attention.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "Leo2 packed forward requires an integer attention mask, "
                f"got sample {sample_index} dtype={attention.dtype}"
            )
        if attention.device != input_ids[sample_index].device:
            raise ValueError(
                "Leo2 packed forward requires attention_mask and input_ids on the same device; "
                f"sample {sample_index} has {attention.device}/{input_ids[sample_index].device}"
            )
        positive_count = int(torch.count_nonzero(attention > 0))
        declared_length = int(attention[0, 0])
        if positive_count != 1 or declared_length != sequence_lengths[sample_index]:
            raise ValueError(
                "Leo2 packed forward requires each source flash_packed mask to declare its exact "
                f"sequence length in attention_mask[0,0]; sample {sample_index} has "
                f"positive_count={positive_count}, declared={declared_length}, "
                f"width={sequence_lengths[sample_index]}"
            )

        scatter_mask = item.get("cond_text_scatter_mask")
        if scatter_mask is None:
            scatter_mask = item.get("text_mask")
        scatter_mask = _expect_row(
            scatter_mask,
            name=f"model_kwargs.cond_text_scatter_mask[{sample_index}]",
            length=sequence_lengths[sample_index],
        )
        text_valid = int(item["cond_text_mask"].bool().sum())
        scatter_count = int(scatter_mask.bool().sum())
        if text_valid != scatter_count:
            raise ValueError(
                "Leo2 packed forward text states and full-sequence text positions differ; "
                f"sample {sample_index} has {text_valid} states and {scatter_count} positions"
            )

    packed_input_ids = torch.cat(input_ids, dim=1)
    total_length = packed_input_ids.shape[1]
    attention_mask = torch.zeros(
        (1, total_length),
        dtype=kwargs[0]["attention_mask"].dtype,
        device=packed_input_ids.device,
    )
    attention_mask[0, : len(blobs)] = torch.tensor(
        sequence_lengths,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    model_kwargs = dict(kwargs[0])
    model_kwargs["attention_mask"] = attention_mask
    model_kwargs["rope_media_info"] = _pack_rope_media_info(kwargs, sequence_lengths=sequence_lengths, offsets=offsets)
    model_kwargs["cond_text_states"], model_kwargs["cond_text_mask"] = _pack_text_conditions(kwargs)
    for key in _FULL_SEQUENCE_MASKS:
        model_kwargs[key] = _pack_optional_rows(kwargs, key=key, sequence_lengths=sequence_lengths)
    for key in _UNSUPPORTED_PACKED_FIELDS:
        model_kwargs[key] = None

    packing_kwargs: dict[str, Any] = {
        "sample_offsets": torch.tensor(offsets, dtype=torch.long, device="cpu").unsqueeze(0),
    }
    branch_indices: list[torch.Tensor] = []
    for key in _TOKEN_INDEX_FIELDS:
        packed_indices, token_lengths = _pack_token_indices(
            kwargs,
            key=key,
            sequence_lengths=sequence_lengths,
            offsets=offsets,
        )
        model_kwargs[key] = packed_indices
        packing_kwargs[key.replace("indices", "lengths")] = token_lengths
        if packed_indices is not None:
            branch_indices.append(packed_indices)

    if branch_indices:
        partition = torch.cat(branch_indices, dim=1).sort(dim=1).values
        expected = torch.arange(total_length, device=partition.device, dtype=partition.dtype).unsqueeze(0)
        if partition.shape != expected.shape or not torch.equal(partition, expected):
            raise ValueError(
                "Leo2 packed forward requires und/gen/audio token indices to partition every input token exactly once"
            )

    # Pad each branch independently so CP reduce-scatter receives equal sequence lengths.
    pad_count: dict[str, int] = {"und": 0, "gen": 0, "audio": 0}
    next_pad_index = total_length
    for branch, index_key in (
        ("und", "und_token_indices"),
        ("gen", "gen_token_indices"),
        ("audio", "audio_token_indices"),
    ):
        indices = model_kwargs[index_key]
        lengths_key = f"{branch}_token_lengths"
        lengths = packing_kwargs[lengths_key]
        if indices is None:
            continue
        count = (-indices.shape[1]) % sequence_parallel_size
        pad_count[branch] = count
        if count == 0:
            continue
        padding_indices = torch.arange(
            next_pad_index,
            next_pad_index + count,
            dtype=indices.dtype,
            device=indices.device,
        ).unsqueeze(0)
        model_kwargs[index_key] = torch.cat([indices, padding_indices], dim=1)
        lengths = lengths.clone()
        lengths[-1] += count
        packing_kwargs[lengths_key] = lengths
        next_pad_index += count

    padding_length = next_pad_index - total_length
    if padding_length:
        packed_input_ids = torch.cat(
            [
                packed_input_ids,
                torch.zeros(
                    (1, padding_length),
                    dtype=packed_input_ids.dtype,
                    device=packed_input_ids.device,
                ),
            ],
            dim=1,
        )
        model_kwargs["attention_mask"] = torch.cat(
            [
                model_kwargs["attention_mask"],
                torch.zeros(
                    (1, padding_length),
                    dtype=model_kwargs["attention_mask"].dtype,
                    device=model_kwargs["attention_mask"].device,
                ),
            ],
            dim=1,
        )
        for key in _FULL_SEQUENCE_MASKS:
            row = model_kwargs[key]
            if row is not None:
                model_kwargs[key] = torch.cat(
                    [
                        row,
                        torch.zeros(
                            (1, padding_length),
                            dtype=row.dtype,
                            device=row.device,
                        ),
                    ],
                    dim=1,
                )

    packing_kwargs["pad_count"] = pad_count
    padded_branch_indices = [model_kwargs[key] for key in _TOKEN_INDEX_FIELDS if model_kwargs[key] is not None]
    if padded_branch_indices:
        partition = torch.cat(padded_branch_indices, dim=1).sort(dim=1).values
        expected = torch.arange(next_pad_index, device=partition.device, dtype=partition.dtype).unsqueeze(0)
        if partition.shape != expected.shape or not torch.equal(partition, expected):
            raise AssertionError("Leo2 internal branch padding did not preserve the token partition")

    return PackedLeo2Conditions(
        input_ids=packed_input_ids,
        model_kwargs=model_kwargs,
        packing_kwargs=packing_kwargs,
        batch_size=len(blobs),
        sequence_lengths=sequence_lengths,
    )


def as_packed_media(
    batch: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor | list[torch.Tensor]]:
    """Convert dense or shape-ragged media to one native physical sequence row."""

    if isinstance(batch, torch.Tensor):
        if batch.ndim < 3 or batch.shape[0] < 1:
            raise ValueError("Leo2 packed media requires a non-empty batched Tensor")
        # Keep one physical row while retaining logical B for the patch projection.
        return [batch]
    if not isinstance(batch, (list, tuple)) or not batch:
        raise ValueError("Leo2 ragged packed media requires a non-empty sequence of Tensors")
    samples = list(batch)
    if any(not isinstance(sample, torch.Tensor) for sample in samples):
        raise TypeError("Leo2 ragged packed media samples must all be Tensors")
    if any(sample.ndim < 3 or sample.shape[0] != 1 for sample in samples):
        raise ValueError(
            "Leo2 ragged packed media expects per-sample Tensors with leading dimension one; "
            f"got {[tuple(sample.shape) for sample in samples]}"
        )
    first = samples[0]
    if any(sample.device != first.device or sample.dtype != first.dtype for sample in samples[1:]):
        raise ValueError("Leo2 ragged packed media samples must share device and dtype")
    # Native BatchRaggedMedia represents a physical row as a list of unbatched media.
    return [[sample.squeeze(0) for sample in samples]]


def unpack_packed_prediction(
    prediction: Any,
    *,
    batch_size: int,
    name: str,
) -> torch.Tensor | list[torch.Tensor]:
    """Restore a packed output, retaining a list when logical shapes differ."""

    if isinstance(prediction, torch.Tensor):
        if prediction.shape[0] != batch_size:
            raise ValueError(f"Leo2 {name} prediction expected batch {batch_size}, got {tuple(prediction.shape)}")
        return prediction
    if (
        not isinstance(prediction, (list, tuple))
        or len(prediction) != 1
        or not isinstance(prediction[0], (list, tuple))
        or len(prediction[0]) != batch_size
    ):
        raise TypeError(
            f"Leo2 packed {name} prediction expected [[{batch_size} tensors]], got {type(prediction).__name__}"
        )
    samples = list(prediction[0])
    if any(not isinstance(sample, torch.Tensor) for sample in samples):
        raise TypeError(f"Leo2 packed {name} prediction contains a non-Tensor sample")
    if any(sample.ndim == 0 or sample.shape[0] != 1 for sample in samples):
        raise ValueError(f"Leo2 packed {name} predictions must have leading dimension one")
    reference_shape = samples[0].shape[1:]
    if all(sample.shape[1:] == reference_shape for sample in samples):
        return torch.cat(samples, dim=0)
    return samples


__all__ = [
    "PackedLeo2Conditions",
    "as_packed_media",
    "pack_hymm_conditions",
    "unpack_packed_prediction",
]
