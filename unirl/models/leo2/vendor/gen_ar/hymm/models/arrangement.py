# ======================================================================
# Definition of token vocabulary arrangement and sequence arrangement
#

from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class Arrangement:
    # Vocabulary arrangement (contains `_id`):
    text_id_range: Tuple[int, int]
    pad_id: int
    img_id: int
    mask_id: int
    uncond_id: int
    boi_id: int
    eoi_id: int

    # Sequence arrangement (starts with `s_`):
    #     When sequence is dynamic, ranges could be undefined.
    s_text_range: Tuple[Optional[int], Optional[int]]
    s_text_maxlen: int
    s_image_range: Tuple[Optional[int], Optional[int]]
    s_image_maxlen: int
    sequence: List[str]

    image_id_range: Tuple[int, int] = (-1, -1)
    ignore_id: int = -100

    def __repr__(self):
        str_ = "    Token vocabulary:"
        vocab_repr_list = ['']
        for key, value in vars(self).items():
            if "_id" in key:
                name = key.split('_')[0]
                if 'range' in key:
                    vocab_repr_list.append(f"{name} token: {value[0]} ~ {value[1]} ({value[1] - value[0]})")
                else:
                    vocab_repr_list.append(f"{name} token: {value}")
        str_ += "\n        ".join(vocab_repr_list)

        str_ += f"\n    Token sequence: [{' '.join(self.sequence)}]"
        seq_repr_list = ['']
        for token in self.sequence:
            name = token.strip("<|>")
            if hasattr(self, f"s_{name}_range"):
                value = getattr(self, f"s_{name}_range")
                max_value = getattr(self, f"s_{name}_maxlen")
                seq_repr_list.append(f"{name} token: {value[0]} ~ {value[1]} (max={max_value})")
            else:
                seq_repr_list.append(f"{name} token: {token}")
        str_ += "\n        ".join(seq_repr_list)

        return str_
