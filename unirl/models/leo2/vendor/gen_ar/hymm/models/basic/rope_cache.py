import copy

import torch

from .pos_emb_layers import (
    get_batch_text_image_2d_rope,
    get_batch_xdrope,
    get_batch_text_media_3d_rope,
    get_batch_leo2_3d_rope,
    get_batch_interleaved_mrope,
)


class CachedRoPE(object):
    """Cache RoPE tables by every argument that affects their construction."""

    def __init__(self, config):
        self._config = config
        self.cos_cache = None
        self.sin_cache = None
        self.seq_len = None
        self.rope_media_info = None
        self._cache_signature = None
        self.cache_hits = 0
        self.cache_misses = 0
        print(f"Using cached RoPE: {self._config.rope_type}")

    @staticmethod
    def _freeze_cache_value(value):
        if isinstance(value, torch.Tensor):
            return (
                "tensor",
                str(value.device),
                str(value.dtype),
                tuple(value.shape),
                tuple(value.detach().cpu().reshape(-1).tolist()),
            )
        if isinstance(value, slice):
            return ("slice", value.start, value.stop, value.step)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (
                        (CachedRoPE._freeze_cache_value(key), CachedRoPE._freeze_cache_value(item))
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(CachedRoPE._freeze_cache_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted((CachedRoPE._freeze_cache_value(item) for item in value), key=repr))
        return value

    def _make_cache_signature(self, seq_len, device, rope_media_info, sample_offsets):
        return (
            int(seq_len),
            str(torch.device(device)),
            self._freeze_cache_value(rope_media_info),
            self._freeze_cache_value(sample_offsets),
        )

    def reset(self):
        self.cos_cache = None
        self.sin_cache = None
        self.seq_len = None
        self.rope_media_info = None
        self._cache_signature = None

    @torch.autocast(device_type='cuda', enabled=False)
    def __call__(self, seq_len, device, rope_media_info=None, input_pos=None, sample_offsets=None):
        """Return cached RoPE tables, slicing them with input_pos when requested."""
        cache_signature = self._make_cache_signature(seq_len, device, rope_media_info, sample_offsets)
        if self._cache_signature != cache_signature:
            # Cache miss, compute RoPE
            rope_media_info_for_compute = copy.deepcopy(rope_media_info)
            if self._config.rope_type in ["2d", "default"]:
                cos_cache, sin_cache = get_batch_text_image_2d_rope(
                    image_infos=rope_media_info_for_compute,
                    seq_len=seq_len,
                    n_elem=self._config.attention_head_size,
                    device=device,
                    base=self._config.rope_theta,
                    base_rescale_factor=self._config.rope_scaling,
                    sample_offsets=sample_offsets
                )
            elif self._config.rope_type == "interleaved_mrope":
                cos_cache, sin_cache = get_batch_interleaved_mrope(
                    image_infos=rope_media_info_for_compute,
                    seq_len=seq_len,
                    mrope_section=self._config.mrope_section,
                    n_elem=self._config.attention_head_size,
                    device=device,
                    base=self._config.rope_theta,
                    base_rescale_factor=self._config.rope_scaling,
                    sample_offsets=sample_offsets
                )
            elif self._config.rope_type == "xdrope":
                cos_cache, sin_cache = get_batch_xdrope(
                    image_infos=rope_media_info_for_compute,
                    seq_len=seq_len,
                    n_elem=self._config.attention_head_size,
                    xdrope_section=self._config.xdrope_section,
                    device=device,
                    base=self._config.rope_theta,
                    base_rescale_factor=self._config.rope_scaling,
                    sample_offsets=sample_offsets
                )
            elif self._config.rope_type == "3d":
                cos_cache, sin_cache = get_batch_text_media_3d_rope(
                    media_infos=rope_media_info_for_compute,
                    seq_len=seq_len,
                    n_elem=self._config.attention_head_size,
                    mrope_section=self._config.mrope_section,
                    device=device,
                    base=self._config.rope_theta,
                    base_rescale_factor=self._config.rope_scaling,
                    sample_offsets=sample_offsets,
                    float_position=self._config.rope_float_position,
                    no_space=self._config.rope_no_space,
                    fixed_space=self._config.rope_fixed_space,
                    cond_image_fixed_space=self._config.rope_cond_image_fixed_space,
                    cond_video_fixed_space=self._config.rope_cond_video_fixed_space
                )
            elif self._config.rope_type == "leo2_3d":
                cos_cache, sin_cache = get_batch_leo2_3d_rope(
                    media_infos=rope_media_info_for_compute,
                    seq_len=seq_len,
                    n_elem=self._config.attention_head_size,
                    mrope_section=self._config.mrope_section,
                    device=device,
                    base=self._config.rope_theta,
                    base_rescale_factor=self._config.rope_scaling,
                    use_scale_rope=self._config.use_scale_rope,
                )
            else:
                raise NotImplementedError(f"rope_type `{self._config.rope_type}` not supported")

            # Do not publish a new key until all cache tensors were built successfully.
            self.cos_cache, self.sin_cache = cos_cache, sin_cache
            self.seq_len = seq_len
            self.rope_media_info = rope_media_info
            self._cache_signature = cache_signature
            self.cache_misses += 1
        else:
            self.cache_hits += 1

        if input_pos is None:
            # Typically for training
            cos, sin = self.cos_cache, self.sin_cache
        else:
            # Typically for inference
            assert input_pos.dim() == 2, f"{input_pos.shape=}"
            head_size = self.cos_cache.size(-1)
            cos = torch.gather(self.cos_cache, dim=1, index=input_pos.unsqueeze(-1).expand(-1, -1, head_size))
            sin = torch.gather(self.sin_cache, dim=1, index=input_pos.unsqueeze(-1).expand(-1, -1, head_size))

        return cos, sin
