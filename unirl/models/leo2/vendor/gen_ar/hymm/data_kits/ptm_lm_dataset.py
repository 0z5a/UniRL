# Copyright (c) 2025. LM dataset for bin/idx format using AngelPTM (Megatron) GPTDataset.
# Allows loading pretokenized .bin/.idx LM data via core_gpt_dataset_config_from_args logic.

from __future__ import annotations

from loguru import logger as all_logger
from typing import Any, Dict, List, Optional, Tuple

import torch



def _get_megatron_tokenizer(tokenizer_path: str):
    """Initialize Megatron HuggingFace tokenizer (same as AngelPTM toolkits/dataset.py)."""
    from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer
    return _HuggingFaceTokenizer(tokenizer_path, trust_remote_code=True)


def _build_gpt_dataset_config(
    data_path: str,
    split: str,
    sequence_length: int,
    tokenizer: Any,
    seed: int = 1234,
    create_attention_mask: bool = False,
    low_level_data_type: str = "native",
    path_to_cache: Optional[str] = None,
    mmap_bin_files: bool = True,
    sft_eod_id: int = -1,
    sft_pad_id: int = -1,
    reset_position_ids: bool = True,
    reset_attention_mask: bool = True,
    **kwargs: Any,
) -> Any:
    """Build GPTDatasetConfig and train GPT dataset using AngelPTM/Megatron builders."""
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder

    blend: Tuple[List[str], Optional[List[float]]] = ([data_path], None)
    config = GPTDatasetConfig(
        random_seed=seed,
        sequence_length=sequence_length,
        blend=blend,
        blend_per_split=None,
        split=split,
        multiple_validation_sets=False,
        full_validation=False,
        num_dataset_builder_threads=1,
        path_to_cache=path_to_cache,
        mmap_bin_files=mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=reset_position_ids,
        reset_attention_mask=reset_attention_mask,
        eod_mask_loss=False,
        create_attention_mask=create_attention_mask,
        create_document_segments=False,
        add_extra_token_to_sequence=True,
        object_storage_cache_path=None,
        mid_level_dataset_surplus=0.005,
        sft_eod_id=sft_eod_id if sft_eod_id != -1 else tokenizer.eod,
        sft_pad_id=sft_pad_id if sft_pad_id != -1 else tokenizer.pad,
        load_from_legacy_index_path=False,
        doc_idx_shuffle_strategy=None,
        shuffle_idx_shuffle_strategy=None,
        low_level_data_type=low_level_data_type,
        **kwargs,
    )

    # Train only: sizes = [None, 0, 0] -> train_ds has one epoch, no valid/test
    sizes: List[Optional[int]] = [None, 0, 0]

    def is_built_on_rank() -> bool:
        return True

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        sizes,
        is_built_on_rank,
        config,
    ).build()

    return train_ds


class PTMLMDatasetWrapper(torch.utils.data.Dataset):
    """Wraps Megatron GPTDataset so that __getitem__ returns the batch format expected by
    prepare_model_lm_inputs (tokens, target_tokens, text_mask, dataset_tag, dummy_type_dict, etc.).
    """

    def __init__(
        self,
        gpt_dataset: Any,
        dataset_tag: str,
        dummy_type_dict: Optional[Dict[str, int]] = None,
        truncation_mask_id: int = -1,
        bos_mask_loss: bool = False,
    ):
        self.gpt_dataset = gpt_dataset
        self.dataset_tag = dataset_tag
        self.dummy_type_dict = dummy_type_dict or {}

        self.truncation_mask_id = truncation_mask_id
        self.bos_mask_loss = bos_mask_loss

    def __len__(self) -> int:
        return len(self.gpt_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.gpt_dataset[idx]
        # GPTDataset returns: tokens (seqlen), labels (seqlen), loss_mask (seqlen), position_ids, (attention_mask)
        tokens = sample["tokens"]  # (seqlen,) already input side; labels are targets
        labels = sample["labels"]
        loss_mask = sample["loss_mask"]

        if self.truncation_mask_id != -1:
            loss_mask[labels == self.truncation_mask_id] = 0.0

        if self.bos_mask_loss:
            loss_mask[labels == self.gpt_dataset.config.tokenizer.bos] = 0.0

        # prepare_model_lm_inputs uses tokens[:, :-1] and target_tokens[:, 1:], so we need
        # one extra position for the shift. GPTDataset already gives tokens = input, labels = target
        # with same length (add_extra_token_to_sequence). So tokens is [0..n-1], labels is [1..n].
        # Our batch format: "tokens" = full sequence (input), "target_tokens" = labels with -100 where mask=0
        seqlen = tokens.shape[0]
        target_tokens = labels.clone()
        target_tokens[loss_mask == 0.0] = -100
        text_mask = loss_mask.float()

        bos_id = self.gpt_dataset.config.tokenizer.bos
        bos_positions = (tokens == bos_id).nonzero(as_tuple=True)[0]
        if bos_positions.numel() == 0:
            lengths = [seqlen]
            offsets = [0, seqlen]
        else:
            # 按 BOS 位置分割：段边界 = [bos_0, bos_1, ..., bos_{n-1}, seqlen]，段 i = [offsets[i], offsets[i+1])
            end_positions = torch.cat([
                bos_positions[1:],
                torch.tensor([seqlen], device=tokens.device, dtype=bos_positions.dtype),
            ])
            lengths = (end_positions - bos_positions).tolist()
            offsets = list(bos_positions.cpu().tolist()) + [seqlen]
        n_samples = len(lengths)

        return {
            "dataset_tag": self.dataset_tag,
            "n_samples": n_samples,
            "index": idx,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "offsets": torch.tensor(offsets, dtype=torch.long, device=tokens.device),
            "dummy_type_dict": self.dummy_type_dict,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        tokens = torch.stack([b["tokens"] for b in batch])
        target_tokens = torch.stack([b["target_tokens"] for b in batch])
        text_mask = torch.stack([b["text_mask"] for b in batch])

        ret = {
            "dataset_tag": [b["dataset_tag"] for b in batch],
            "n_samples": torch.tensor([b["n_samples"] for b in batch]),
            "index": [b["index"] for b in batch],
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "dummy_type_dict": [b["dummy_type_dict"] for b in batch],
        }
        if "offsets" in batch[0]:
            ret["offsets"] = [b["offsets"] for b in batch]
        return ret


def build_ptm_lm_dataset(
    args,
    dataset_tag,
    task_kwargs,
    index_kwargs=None,
    logger=None,
) -> PTMLMDatasetWrapper:
    """Build LM dataset from bin/idx using AngelPTM GPTDataset + BlendedMegatronDatasetBuilder.

    task_kwargs (lm_ptm_task_kwargs) should contain:
        - data_path: path to bin/idx prefix (or use args.DATA_PATH). For multibin can be:
            - str: single path or comma-separated paths, or range "/path/prefix_{0...99}"
            - list: multiple paths, e.g. ["/path/to/d_0", "/path/to/d_1", ...]
        - split: e.g. "100,0,0"
        - seq_length or max_token_length: sequence length
    Optional: seed, low_level_data_type ("native" | "multibin"), path_to_cache.
    """
    log = logger or all_logger
    ptm_kwargs = task_kwargs or {}
    data_path = ptm_kwargs.get("data_path")
    # Megatron handle_multibin_path expects str (comma-separated); support YAML list by joining
    if isinstance(data_path, (list, tuple)):
        data_path = ",".join(str(p).strip() for p in data_path)

    split = ptm_kwargs.get("split", "100,0,0")
    seq_length = ptm_kwargs.get("seq_length") or ptm_kwargs.get("max_token_length")  # ptm lm dataset is fixed length
    seed = ptm_kwargs.get("shuffle_seed", getattr(args, "seed", 1234))
    low_level_data_type = ptm_kwargs.get("low_level_data_type", "native")
    path_to_cache = ptm_kwargs.get("path_to_cache")
    mmap_bin_files = ptm_kwargs.get("mmap_bin_files", True)
    tokenizer_name = ptm_kwargs.get("tokenizer_name")

    # Megatron tokenizer
    from hymm.constants import TOKENIZER_PATH
    if tokenizer_name in TOKENIZER_PATH:
        tokenizer_name = TOKENIZER_PATH[tokenizer_name]
    tokenizer = _get_megatron_tokenizer(tokenizer_name)
    sft_eod_id = ptm_kwargs.get("sft_eod_id", -1)
    sft_pad_id = ptm_kwargs.get("sft_pad_id", -1)
    truncation_mask_id = ptm_kwargs.get("truncation_mask_id", -1)
    reset_position_ids = ptm_kwargs.get("reset_position_ids", False)
    reset_attention_mask = ptm_kwargs.get("reset_attention_mask", False)
    bos_mask_loss = ptm_kwargs.get("bos_mask_loss", False)
    create_attention_mask = not ptm_kwargs.get("no_create_attention_mask_in_dataloader", False)


    log.info(f"[PTMLMDataset] Building LM bin/idx dataset: data_path={data_path}, split={split}, seq_length={seq_length}")

    gpt_train_ds = _build_gpt_dataset_config(
        data_path=data_path,
        split=split,
        sequence_length=seq_length,
        tokenizer=tokenizer,
        seed=seed,
        create_attention_mask=create_attention_mask,
        low_level_data_type=low_level_data_type,
        path_to_cache=path_to_cache,
        mmap_bin_files=mmap_bin_files,
        sft_eod_id=sft_eod_id,
        sft_pad_id=sft_pad_id,
        reset_position_ids=reset_position_ids,
        reset_attention_mask=reset_attention_mask,
    )

    if gpt_train_ds is None:
        raise RuntimeError("BlendedMegatronDatasetBuilder returned None for train split. Check data_path and split.")

    # Dummy tokens for pure LM: no image/vit dummies unless args say otherwise
    dummy_type_dict = {}
    if getattr(args, "use_vit", False):
        from ..constants import VISION_ENCODER_META_INFO
        vit_type = getattr(args, "vit_type", None)
        if vit_type and vit_type in VISION_ENCODER_META_INFO:
            dummy_type_dict["vit"] = VISION_ENCODER_META_INFO[vit_type].get("dummy_number", 0)
    if getattr(args, "use_vae", False) and getattr(args, "add_timestep_token", False):
        dummy_type_dict["ts_proj"] = 1 + (1 if getattr(args, "add_timestep_token", False) else 0)

    wrapper = PTMLMDatasetWrapper(
        gpt_dataset=gpt_train_ds,
        dataset_tag=dataset_tag,
        dummy_type_dict=dummy_type_dict,
        truncation_mask_id=truncation_mask_id,
        bos_mask_loss=bos_mask_loss,
    )
    log.info(f"[PTMLMDataset] Built PTM LM dataset with {len(wrapper)} samples.")
    return wrapper
