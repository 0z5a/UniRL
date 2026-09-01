"""
Monkey-patch Qwen3.5 transformers model to support cu_seqlens packed sequence inference.

Eliminates padding waste in text encoder forward when samples have variable lengths.
This patch adds cu_seqlens support to three classes:
  - Qwen3_5GatedDeltaNet.forward  (conv1d seq_idx + chunk_gated_delta_rule cu_seqlens)
  - Qwen3_5DecoderLayer.forward   (thread cu_seqlens/seq_idx through)
  - Qwen3_5TextModel.forward      (packed path: searchsorted position_ids, skip causal_mask)

All patches are backward-compatible: when cu_seqlens is None, behavior is identical to vanilla.

It also provides patch_fla_l2norm_disable_autotune(), which pins fla's l2norm forward
kernels to a fixed config to avoid runtime autotune latency spikes during inference.

Usage:
    from hymm.models.text_encoder.patch_qwen3_5_packed import patch_qwen3_5_for_packed_sequence
    patch_qwen3_5_for_packed_sequence()  # call once after transformers is imported
"""

import sys

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Patch 1: Qwen3_5GatedDeltaNet.forward
# ---------------------------------------------------------------------------


def _patched_gated_delta_net_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    seq_idx=None,
):
    """
    Patched GatedDeltaNet forward with cu_seqlens support for packed sequences.

    New params:
        cu_seqlens: [N+1] int32 tensor of cumulative sequence lengths
        cu_seqlens_cpu: pre-computed CPU copy (avoids per-layer D2H sync)
        seq_idx: [1, total_len] int32 tensor mapping each position to its sequence index
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    # When cu_seqlens is provided, input is packed (no padding) — skip masking
    if cu_seqlens is None:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

    if use_precomputed_states:
        conv_state = cache_params.layers[self.layer_idx].conv_states
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # Build seq_idx for causal_conv1d when using packed sequences
    _conv_seq_idx = None
    if cu_seqlens is not None and self.causal_conv1d_fn is not None:
        if seq_idx is not None:
            # Pre-computed seq_idx passed from outside (no D2H sync)
            _conv_seq_idx = seq_idx
        else:
            # Construct seq_idx from cu_seqlens using searchsorted (pure GPU, no D2H)
            # Use int64 explicitly to guarantee searchsorted dtype compatibility
            positions = torch.arange(seq_len, device=cu_seqlens.device, dtype=torch.int64)
            _conv_seq_idx = (
                torch.searchsorted(cu_seqlens.to(torch.int64), positions, right=True) - 1
            ).to(torch.int32).unsqueeze(0)

    if use_precomputed_states and seq_len == 1:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if use_precomputed_states:
            mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
        if cache_params is not None:
            new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.update_conv_state(new_conv_state, self.layer_idx)
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=_conv_seq_idx,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
        if use_precomputed_states:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if use_precomputed_states and seq_len == 1:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        # Build cu_seqlens kwargs for fla's chunk_gated_delta_rule
        _chunk_kwargs = {}
        if cu_seqlens is not None:
            _chunk_kwargs["cu_seqlens"] = cu_seqlens.to(torch.int32)
            _chunk_kwargs["cu_seqlens_cpu"] = (
                cu_seqlens_cpu if cu_seqlens_cpu is not None
                else cu_seqlens.cpu().to(torch.int32)
            )

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=recurrent_state if use_precomputed_states else None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            **_chunk_kwargs,
        )

    # Update cache
    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = self.norm(core_attn_out, z)
    return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


# ---------------------------------------------------------------------------
# Patch 2: Qwen3_5DecoderLayer.forward
# ---------------------------------------------------------------------------


def _patched_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    seq_idx=None,
    **kwargs,
):
    """
    Patched DecoderLayer forward that threads cu_seqlens/seq_idx to linear_attn sublayer.
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Token Mixer
    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            seq_idx=seq_idx,
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


# ---------------------------------------------------------------------------
# Patch 3: Qwen3_5TextModel.forward
# ---------------------------------------------------------------------------

# We store a reference to the original (decorated) forward so we can delegate
# the non-packed path without losing @merge_with_config_defaults / @capture_outputs.
_original_text_model_forward = None


def _patched_text_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    cu_seqlens=None,
    **kwargs,
):
    """
    Patched TextModel forward. When cu_seqlens is provided, runs packed sequence path.
    Otherwise delegates to the original forward (preserving its decorators).
    """
    # --- Non-packed path: delegate to original decorated forward ---
    if cu_seqlens is None:
        return _original_text_model_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    # --- Packed sequence path ---
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast
    from transformers.cache_utils import DynamicCache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    # Build position_ids that reset at each sequence boundary
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    total_len = inputs_embeds.shape[1]

    # Pre-compute seq_indices (used for both position_ids and seq_idx)
    positions = torch.arange(total_len, device=inputs_embeds.device, dtype=cu_seqlens.dtype)
    seq_indices = torch.searchsorted(cu_seqlens, positions, right=True) - 1

    if position_ids is None:
        # Pure GPU: positions[i] = i - cu_seqlens[seq_of(i)], no D2H sync
        per_seq_pos = positions - cu_seqlens[seq_indices]
        # Shape: [4, 1, total_len] for the 4-way position encoding (text, temporal, height, width)
        position_ids = per_seq_pos.view(1, 1, -1).expand(4, 1, -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    # No causal mask needed — flash_attn_varlen handles boundaries via cu_seqlens
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # Compute max_seqlen for flash attention (single D2H for this scalar is unavoidable)
    max_seqlen = int(lengths.max().item())

    # Pre-compute seq_idx ONCE for all linear_attention layers (pure GPU, no D2H sync)
    seq_idx = seq_indices.to(torch.int32).unsqueeze(0)  # [1, total_len]

    # Pre-compute cu_seqlens_cpu ONCE (single D2H), reused across all layers
    cu_seqlens_i32 = cu_seqlens.to(torch.int32)
    cu_seqlens_cpu = cu_seqlens_i32.cpu()

    # Check if we need to collect hidden states (for hidden_state_skip_layer)
    output_hidden_states = kwargs.pop("output_hidden_states", False)
    all_hidden_states = () if output_hidden_states else None

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.config.layer_types[i] == "linear_attention":
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cu_seqlens=cu_seqlens_i32,
                cu_seqlens_cpu=cu_seqlens_cpu,
                seq_idx=seq_idx,
                **kwargs,
            )
        else:
            # Full attention: pass cu_seq_lens for flash_attn_varlen
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cu_seqlens=None,  # not used by full_attention directly
                cu_seq_lens_q=cu_seqlens_i32,
                cu_seq_lens_k=cu_seqlens_i32,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                **kwargs,
            )

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
    )


# ---------------------------------------------------------------------------
# Patch 4: fla l2norm forward — pin kernel config (disable runtime autotune)
# ---------------------------------------------------------------------------


def _is_triton_autotuner(x):
    """True if ``x`` is a triton Autotuner OR any subclass of it.
    """
    try:
        from triton.runtime.autotuner import Autotuner as _TritonAutotuner
        if isinstance(x, _TritonAutotuner):
            return True
    except Exception:  # noqa: BLE001
        pass
    return type(x).__name__.endswith("Autotuner") and hasattr(x, "configs") and hasattr(x, "fn")


def patch_fla_l2norm_disable_autotune(fwd_bt: int = 64, fwd_warps: int = 4, logger=None):
    """
    Replace fla's autotuned l2norm forward kernels with a fixed-config launch.

    The text encoder runs l2norm inference-only on variable packed sequence lengths.
    fla's stock kernels are wrapped in @triton.autotune, which re-benchmarks (do_bench
    over many configs) on every new D/NB key — causing multi-second latency spikes.
    Here we unwrap the autotuner to its base @triton.jit kernel and launch it with a
    pinned BT / num_warps instead.

    `l2norm_fwd` is re-bound everywhere it was imported by name (e.g. fla.ops.*.chunk,
    which call it on the hot path via use_qk_l2norm_in_kernel=True), so already-loaded
    modules pick up the pinned version too. Idempotent.
    """
    try:
        import triton
        import fla.modules.l2norm as _l2
    except ImportError:
        return False

    if getattr(_l2, "_l2norm_autotune_pinned", False):
        return False

    orig_fwd = _l2.l2norm_fwd

    def _unwrap_kernel(k):
        # Unwrap all autotuner layers (incl. fla's CachedAutotuner subclass) down to
        # the raw @triton.jit function, so we can launch with a fixed BT/num_warps
        # instead of entering the autotuner's benchmarking path.
        while _is_triton_autotuner(k):
            k = k.fn
        return k

    base_kernel = _unwrap_kernel(_l2.l2norm_fwd_kernel)
    base_kernel1 = _unwrap_kernel(_l2.l2norm_fwd_kernel1)

    def l2norm_fwd(x, eps=1e-6, output_dtype=None):
        x_shape_og = x.shape
        x = x.view(-1, x.shape[-1])
        y = torch.empty_like(x) if output_dtype is None else torch.empty_like(x, dtype=output_dtype)
        assert y.stride(-1) == 1
        T, D = x.shape[0], x.shape[-1]
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
        if D > BD:
            raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
        rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
        if D <= 512:
            NB = triton.cdiv(T, 2048 * 32)
            grid = (triton.cdiv(T, fwd_bt),)
            base_kernel[grid](
                x=x, y=y, rstd=rstd, eps=eps, T=T, D=D, BD=BD, NB=NB,
                BT=fwd_bt, num_warps=fwd_warps,
            )
        else:
            base_kernel1[(T,)](x=x, y=y, rstd=rstd, eps=eps, D=D, BD=BD, num_warps=fwd_warps)
        return y.view(x_shape_og), rstd.view(x_shape_og[:-1])

    # Re-bind every reference to the original function (defining module + by-name imports).
    # Only `fla.*` modules import l2norm_fwd by name; inspect __dict__ directly so we never
    # trigger lazy __getattr__ machinery (e.g. transformers _LazyModule) on other packages.
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("fla"):
            continue
        mod_dict = getattr(mod, "__dict__", None)
        if mod_dict is not None and mod_dict.get("l2norm_fwd", None) is orig_fwd:
            mod_dict["l2norm_fwd"] = l2norm_fwd

    _l2._l2norm_autotune_pinned = True
    if logger is not None:
        logger.info(f"Pinned fla l2norm forward (BT={fwd_bt}, num_warps={fwd_warps}); autotune disabled.")
    return True


# ---------------------------------------------------------------------------
# Patch 5: disable ALL fla triton autotune on the encoder's hot path
# ---------------------------------------------------------------------------


def _unwrap_to_autotuner(obj):
    """Walk the triton decorator chain (Heuristics -> Autotuner -> JITFunction)
    via ``.fn`` and return the first Autotuner found, else None.
    """
    seen = set()
    cur = obj
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _is_triton_autotuner(cur):
            return cur
        cur = getattr(cur, "fn", None)
    return None


def _pin_autotuner(at):
    """Reduce an Autotuner's config list to ONE deterministic config.
    """
    cfgs = list(getattr(at, "configs", []) or [])
    if len(cfgs) <= 1:
        return False

    def _key(c):
        # Rank the lowest-resource config first to minimize "out of resource"
        # compile failures when forcing a single config across all shapes: order by
        # num_stages (dominant shared-memory driver), then num_warps, then block size.
        kw = getattr(c, "kwargs", {}) or {}
        block_sum = sum(v for v in kw.values() if isinstance(v, int))
        return (
            int(getattr(c, "num_stages", 1) or 1),
            int(getattr(c, "num_warps", 1) or 1),
            block_sum,
            str(kw),
        )

    at.configs = [sorted(cfgs, key=_key)[0]]
    cache = getattr(at, "cache", None)
    if isinstance(cache, dict):
        cache.clear()  # drop any config already selected for a shape key
    return True


def patch_fla_disable_autotune(logger=None, force_import=True):
    """Pin every fla triton autotuner reachable from the Qwen3.5 9B encoder to a
    single fixed config, disabling runtime (timing-based) autotune selection.
    """
    try:
        import triton  # noqa: F401
    except ImportError:
        return False

    # The encoder's training forward (no cache, seq_len>1) goes through
    # chunk_gated_delta_rule + the gated RMSNorm + l2norm. Force-import that
    # reachable set so their Autotuner objects exist before we scan (they are
    # normally imported already by model build time, but be robust to lazy paths).
    if force_import:
        for m in (
            "fla.ops.gated_delta_rule",
            "fla.ops.gated_delta_rule.chunk",
            "fla.modules.fused_norm_gate",
            "fla.modules.layernorm",
            "fla.modules.l2norm",
            "fla.modules.activations",
        ):
            try:
                __import__(m)
            except Exception:
                pass

    pinned = 0
    scanned = 0
    pinned_ids = set()
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "fla" or name.startswith("fla.")):
            continue
        mod_dict = getattr(mod, "__dict__", None)
        if not mod_dict:
            continue
        for obj in list(mod_dict.values()):
            at = _unwrap_to_autotuner(obj)
            if at is None or id(at) in pinned_ids:
                continue
            pinned_ids.add(id(at))
            scanned += 1
            if _pin_autotuner(at):
                pinned += 1

    if logger is not None:
        logger.info(
            f"Disabled fla triton autotune: pinned {pinned}/{scanned} autotuners "
            f"to a single config."
        )
    return pinned > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_packed_sequence_patched():
    """Check if the Qwen3.5 packed sequence patch has been applied."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        return getattr(Qwen3_5GatedDeltaNet, '_packed_sequence_patched', False)
    except ImportError:
        return False


def patch_qwen3_5_for_packed_sequence():
    """
    Monkey-patch Qwen3_5 transformers classes to support cu_seqlens packed sequences.

    Safe to call multiple times (idempotent). Patches are applied at the class level
    so all instances (existing and future) benefit automatically.

    Must be called after transformers is imported (typically right after loading the model).

    Requires: causal-conv1d, flash-linear-attention (fla), flash-attn all installed.
    """
    global _original_text_model_forward

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        Qwen3_5DecoderLayer,
        Qwen3_5TextModel,
    )

    if getattr(Qwen3_5GatedDeltaNet, '_packed_sequence_patched', False):
        return  # already patched

    # Verify required dependencies before patching
    from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
    missing = []
    if not is_causal_conv1d_available():
        missing.append("causal-conv1d (pip install causal-conv1d)")
    if not is_flash_linear_attention_available():
        missing.append("flash-linear-attention (pip install flash-linear-attention)")
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        missing.append("flash-attn (pip install flash-attn --no-build-isolation)")
    if missing:
        raise RuntimeError(
            f"Cannot apply packed sequence patch: missing required dependencies:\n"
            f"  {', '.join(missing)}\n"
            f"Packed sequence encoding requires causal-conv1d (for seq_idx boundary handling), "
            f"fla (for cu_seqlens in chunk_gated_delta_rule), and flash-attn (for flash_attn_varlen_func)."
        )

    # Save the original decorated forward (has @merge_with_config_defaults, @capture_outputs)
    _original_text_model_forward = Qwen3_5TextModel.forward

    # Apply patches
    Qwen3_5GatedDeltaNet.forward = _patched_gated_delta_net_forward
    Qwen3_5DecoderLayer.forward = _patched_decoder_layer_forward
    Qwen3_5TextModel.forward = _patched_text_model_forward

    # Mark as patched
    Qwen3_5GatedDeltaNet._packed_sequence_patched = True

    return True


def maybe_apply_patch(logger=None):
    """
    Conditionally apply packed sequence patch based on args.text_encoder_use_pack.

    - text_encoder_use_pack:   cu_seqlens packed sequence support (default off)
    - text_encoder_fla_l2norm_disable_autotune: pin fla l2norm kernel to skip autotune spikes (default off)
    - text_encoder_fla_disable_autotune: pin ALL fla kernels on the encoder hot path
      to a single fixed config  (default off)
    """
    try:
        from ...core.global_vars import get_args
        args = get_args()
    except (AssertionError, RuntimeError, ImportError):
        return

    if getattr(args, "text_encoder_use_pack", False) and not is_packed_sequence_patched():
        patch_qwen3_5_for_packed_sequence()
        if logger is not None:
            logger.info("Patched Qwen3.5 9B for packed sequence support.")

    # Disable ALL fla autotune Do this first so the dedicated l2norm pin can still override the l2norm launcher.
    if getattr(args, "text_encoder_fla_disable_autotune", False):
        patch_fla_disable_autotune(logger=logger)

    if getattr(args, "text_encoder_fla_l2norm_disable_autotune", False):
        patch_fla_l2norm_disable_autotune(
            fwd_bt=int(getattr(args, "text_encoder_l2norm_fwd_bt", 64)),
            fwd_warps=int(getattr(args, "text_encoder_l2norm_fwd_warps", 4)),
            logger=logger,
        )
