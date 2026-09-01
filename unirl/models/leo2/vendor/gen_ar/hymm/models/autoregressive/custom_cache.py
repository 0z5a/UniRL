from typing import Optional, Dict, Any, Tuple

import torch
from transformers.cache_utils import StaticCache

from hymm.utils.import_utils import is_package_version

IS_TRANSFORMERS_LE_4533 = is_package_version("transformers", "<=", "4.53.3")


class HunyuanStaticCache(StaticCache):
    """
    A custom static cache for multi-modal models that supports dynamic extension of the cache
    and inplace updates of the cache.

    This cache supports batch cache_position updates.
    """
    def __init__(self, *args, **kwargs):
        self.dynamic = kwargs.pop("dynamic", False)
        super().__init__(*args, **kwargs)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        if IS_TRANSFORMERS_LE_4533:
            if self.key_cache[layer_idx].device != key_states.device:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
            k_out = self.key_cache[layer_idx]  # max_batch_size x num_key_value_heads x max_cache_len x head_dim
            v_out = self.value_cache[layer_idx]  # max_batch_size x num_key_value_heads x max_cache_len x head_dim
        else:
            if self.layers[layer_idx].keys is None:
                self.layers[layer_idx].lazy_initialization(value_states)
            k_out = self.layers[layer_idx].keys
            v_out = self.layers[layer_idx].values
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            # following steps of gen_text
            if cache_position.dim() == 1:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)

                if self.dynamic:
                    end = cache_position[-1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]
            else:
                # first step of gen_text or all steps of gen_image
                assert cache_position.dim() == 2, f"multiple batch dims not yet {cache_position.shape=}"
                batch_size, idx_size = cache_position.shape
                assert batch_size == k_out.size(0)
                assert batch_size == v_out.size(0)
                assert batch_size == key_states.size(0)
                assert batch_size == value_states.size(0)
                for i in range(batch_size):
                    unbatched_dim = 1
                    k_out[i].index_copy_(unbatched_dim, cache_position[i], key_states[i])
                    v_out[i].index_copy_(unbatched_dim, cache_position[i], value_states[i])

                if self.dynamic:
                    assert len(cache_position) == 1
                    # end = cache_position[0, -1].item() + 1
                    end = int(cache_position[0, -1]) + 1  # Tensor.item()导致图中断
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]

        return k_out, v_out


HunyuanGeminiStaticCache = HunyuanStaticCache   # bc
