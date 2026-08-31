"""Leo2 conditions -- typed container for the diffusion stage.

Unlike SD3/H3, Leo2's per-step forward needs the whole hymm ``model_kwargs``
mapping (input_ids + cond_text_states + attention_mask + media metadata), not
just a text-embed tensor. We carry it as a per-sample list of dicts -- the
same shape ``Part.metadata`` uses -- because everything downstream runs
batch-1 (forward_batch_size=1 / micro_batch_size=1), so list concat is all the
batching these ever need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.conditions import TextEmbedCondition


@dataclass
class Leo2Conditions(Batch):
    """Conditions passed to the Leo2 diffusion stage."""

    # Human-readable / logging-friendly view of the text conditioning.
    text: Optional[TextEmbedCondition] = concat_field(default=None)
    # Per-sample opaque hymm blobs: {"input_ids": Tensor, "model_kwargs": dict}.
    hymm: Optional[List[Dict[str, Any]]] = concat_field(default=None)

    @classmethod
    def from_dict(cls, d: dict) -> "Leo2Conditions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {name: value for name in self.__dataclass_fields__ if (value := getattr(self, name)) is not None}


__all__ = ["Leo2Conditions"]
