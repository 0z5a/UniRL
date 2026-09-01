# Leo backbone with a REAL (non-dummy) second time input -- MeanFlow conditioning.
#
# This is a NEW, self-contained module. It does NOT modify any existing file.
#
# ============================ What this is ============================
# `LeoModelDummyTR` accepts a second time input but ignores it. This model instead
# *uses* the second time `r` exactly the way the reference
# `HunyuanVideo_pureTorch/hymm/models/modules/Leo_t2va.py::LeoT2VABridge_MeanFlow`
# differs from the plain flow-matching `LeoT2VABridge`:
#
#   * Both backbones inject the time via AdaLN-Zero (modulation embedded from a time
#     embedding). The MeanFlow variant builds TWO time embeddings -- one from t and
#     one from r -- and:
#       - the first ~3/4 of the transformer layers do AdaLN with the t embedding,
#       - the last ~1/4 of the layers do AdaLN with the r embedding,
#       - the final (head) layer does AdaLN with the r embedding.
#     (Reference: `index <= meanflow_threshold` -> vec(t), else vec(r); and the head
#     `final_layer(img, video_vec_r)`. With 54 layers, threshold 41 -> 42 t-layers /
#     12 r-layers ~= 3/4 vs 1/4.)
#
# Here `num_layers = 48` (leo-2-moe-v1-1), so the first 36 layers use t and the last
# 12 layers plus both head layers use r. The input token / text embeddings keep using
# t (matching the reference, where img_in/txt_in use t and only AdaLN / head switch).
#
# Math background: improved MeanFlow, https://arxiv.org/pdf/2510.24474 . The network
# learns the average velocity u(z, r, t); conditioning the late layers + head on r lets
# the same backbone represent u(z, r, t) (r==t recovers the instantaneous velocity).
#
# ============================ Backward compatibility ============================
# `forward` accepts `timestep_r` / `audio_timestep_r` (the *model-time* embeddings'
# inputs, i.e. get_model_t(r), matching `timesteps`). When they are None the model
# falls back to using t everywhere, so it behaves identically to the standard
# `LeoModel` -- a safe drop-in. The training/inference loss/return paths are copied
# verbatim from `LeoModel.forward`; only the per-layer / head time-state selection
# changes.
#
# ============================ Selection (no existing file changed) ============================
# Importing this module installs (idempotently) a dispatch extension recognizing
# `model-structure: LeoModelMeanFlow`. Select it on the command line:
#   ... run_pure_torch_leo_imf.sh <config> --model-structure LeoModelMeanFlow

from typing import Optional, Any

import torch

from hymm.core.global_vars import get_parallel_state
from hy_parallelism.context_parallel.core import maybe_scatter_seq, maybe_gather_seq

from .leo import LeoModel, LeoOutput


class LeoModelMeanFlow(LeoModel):
    """`LeoModel` whose late layers + head do AdaLN with a second time input `r`."""

    # ------------------------------------------------------------------ #
    # number of leading layers that keep using t (the rest use r).
    # ------------------------------------------------------------------ #
    def _mf_num_t_layers(self):
        n = int(self._config.num_layers)
        # first 3/4 layers use t; last 1/4 (and the head) use r.
        return (n * 3) // 4

    @staticmethod
    def _mf_time_states(time_embed, timesteps):
        """Build the time-embedding (AdaLN conditioning) for a timestep structure.

        Mirrors how `LeoModel` builds `timestep_states`: a 1-D tensor -> time_embed(t);
        a ragged list -> a list of per-item time_embed(t_i). This matches the `t_emb`
        produced inside `instantiate_vae_media_tokens_full_seqlen`, so the r-states have
        exactly the same structure as the t-states.
        """
        if isinstance(timesteps, (list, tuple)):
            return [time_embed(t_i) for t_i in timesteps]
        return time_embed(timesteps)

    @staticmethod
    def _mf_stack(ts):
        if isinstance(ts, list):
            return torch.stack(ts, dim=0)
        return ts

    # ------------------------------------------------------------------ #
    # Run the tail transformer layers [start_idx, L) with a given per-layer
    # time-state. Used twice: once with r (-> u(t, r)) and once with t (-> u(t, t)).
    # The shared trunk (layers [0, start_idx) with t) is run only once, so the only
    # extra cost of producing the anchor prediction is the last 1/4 layers + head.
    # ------------------------------------------------------------------ #
    def _mf_run_tail(
            self, start_idx, hidden_states, audio_hidden_states, txt_hidden_states,
            vis_layer_ts, aud_layer_ts, attention_mask, cos, sin,
            gen_token_indices, audio_token_indices, und_token_indices,
            gen_token_lengths, audio_token_lengths, und_token_lengths,
    ):
        for layer_idx in range(start_idx, len(self.layers)):
            layer = self.layers[layer_idx]
            if self._audio_config is not None:
                layer_inputs = [
                    (hidden_states, audio_hidden_states, txt_hidden_states),
                    (vis_layer_ts, aud_layer_ts),
                    attention_mask,
                    (cos, sin),
                    (gen_token_indices, audio_token_indices, und_token_indices),
                    (gen_token_lengths, audio_token_lengths, und_token_lengths),
                ]
                hidden_states, audio_hidden_states, txt_hidden_states = layer(*layer_inputs)
            else:
                layer_inputs = [
                    (hidden_states, txt_hidden_states),
                    vis_layer_ts,
                    attention_mask,
                    (cos, sin),
                    (gen_token_indices, und_token_indices),
                    (gen_token_lengths, und_token_lengths),
                ]
                hidden_states, txt_hidden_states = layer(*layer_inputs)
        return hidden_states, audio_hidden_states

    # ------------------------------------------------------------------ #
    # CP gather + scatter-to-full-seqlen + head (AdaLN with the given head
    # time-states). Returns (diff_pred, audio_diff_pred).
    # ------------------------------------------------------------------ #
    def _mf_head(
            self, hidden_states, audio_hidden_states, vis_head_ts, aud_head_ts,
            input_ids, latents, audio_latents, gen_token_indices_, audio_token_indices_,
            visual_mask, audio_mask, rope_media_info, gen_token_lengths, audio_token_lengths, pad_count,
    ):
        if get_parallel_state().cp_size > 1:
            hidden_states = maybe_gather_seq(hidden_states)
            if audio_latents is not None:
                audio_hidden_states = maybe_gather_seq(audio_hidden_states)

        hidden_states_fl = None
        audio_hidden_states_fl = None
        if input_ids is not None:
            if latents is not None:
                hidden_states_fl = torch.zeros(
                    input_ids.size(0), input_ids.size(1), self._config.hidden_size,
                    dtype=hidden_states.dtype, device=hidden_states.device,
                )
                hidden_states_fl.scatter_(dim=1, index=gen_token_indices_.to(hidden_states.device), src=hidden_states)
            if audio_latents is not None:
                audio_hidden_states_fl = torch.zeros(
                    input_ids.size(0), input_ids.size(1), self._audio_config.hidden_size,
                    dtype=audio_hidden_states.dtype, device=audio_hidden_states.device,
                )
                audio_hidden_states_fl.scatter_(
                    dim=1, index=audio_token_indices_.to(audio_hidden_states.device), src=audio_hidden_states
                )

        if latents is not None:
            visual_rope_media_info = [
                [info for info in infos if info[2]['type'] in ["gen_image", "gen_video"]]
                for infos in rope_media_info
            ]
            if input_ids is None:
                diff_pred = self.ragged_final_layer(
                    self.final_layer, hidden_states, vis_head_ts, visual_rope_media_info,
                    token_lengths=gen_token_lengths, pad_count=pad_count["gen"] if pad_count is not None else None,
                )
            else:
                diff_pred = self.ragged_final_layer_full_seqlen(
                    self.final_layer, vis_head_ts, hidden_states_fl, visual_mask, visual_rope_media_info,
                )
        else:
            diff_pred = None
        if audio_latents is not None:
            audio_rope_media_info = [
                [info for info in infos if info[2]['type'] == "gen_audio"]
                for infos in rope_media_info
            ]
            if input_ids is None:
                audio_diff_pred = self.ragged_final_layer(
                    self.audio_final_layer, audio_hidden_states, aud_head_ts, audio_rope_media_info,
                    token_lengths=audio_token_lengths, pad_count=pad_count["audio"] if pad_count is not None else None,
                )
            else:
                audio_diff_pred = self.ragged_final_layer_full_seqlen(
                    self.audio_final_layer, aud_head_ts, audio_hidden_states_fl, audio_mask, audio_rope_media_info,
                )
        else:
            audio_diff_pred = None
        return diff_pred, audio_diff_pred

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            rope_media_info=None,
            return_dict: bool = True,
            # for gen image/video
            latents=None,
            visual_mask=None,
            timesteps=None,
            timesteps_index=None,
            # for gen audio
            audio_latents=None,
            audio_mask=None,
            audio_timesteps=None,
            # for cond text
            cond_text_states=None,
            cond_text_mask=None,
            text_mask=None,
            # sequence packing
            und_token_indices=None,
            gen_token_indices=None,
            audio_token_indices=None,
            sample_offsets=None,
            pad_count=None,
            dummy_count=None,
            und_token_lengths=None,
            gen_token_lengths=None,
            audio_token_lengths=None,
            # only for training
            diffusion_loss_fn: Optional[Any] = None,
            audio_diffusion_loss_fn: Optional[Any] = None,
            visual_loss_weight: float = 1.0,
            audio_loss_weight: float = 1.0,
            repa_feats: Optional[torch.Tensor] = None,
            dataset_tag: str | None = None,
            # only for pipeline parallelism (placeholder)
            ut: Optional[torch.Tensor] = None,
            aut: Optional[torch.Tensor] = None,
            # ===== NEW: second time input (model-time of r). None -> behaves like t. =====
            timestep_r: Optional[torch.Tensor] = None,
            audio_timestep_r: Optional[torch.Tensor] = None,
    ) -> "LeoOutput | tuple":
        _ = ut  # not used for now
        _ = aut  # not used for now
        self.check_types(attention_mask, rope_media_info)
        if latents is None and audio_latents is None:
            raise ValueError("At least one of latents and audio_latents should be provided.")
        # Full-seqlen scatter indices; set inside the input_ids branch, None otherwise.
        gen_token_indices_ = None
        audio_token_indices_ = None

        # === Input layers ===
        assert self._config.patch_size == 1, "instantiate_vae_image_tokens only supports patch_size=1 for now."

        if input_ids is None:
            # Project hidden_states
            timestep_states = self.time_embed(timesteps)
            hidden_states = self.instantiate_vae_media_tokens(
                self.patch_embed, latents, timestep_states, allowed_dims=[4, 5]
            )
            if audio_latents is not None:
                audio_timestep_states = self.audio_time_embed(audio_timesteps)
                audio_hidden_states = self.instantiate_vae_media_tokens(
                    self.audio_projector, audio_latents, audio_timestep_states, allowed_dims=[3, 4]
                )
            else:
                audio_timestep_states = None
                audio_hidden_states = None

            # Project text conditions
            if self._config.text_proj_type == "linear":
                if gen_token_indices is not None and self.training:
                    cond_text_states = cond_text_states.reshape(-1, cond_text_states.size(-1))[
                        cond_text_mask.view(-1).bool()].unsqueeze(0)
                txt_hidden_states = self.text_projector(cond_text_states)
            elif self._config.text_proj_type == "single_refiner":
                txt_hidden_states = self.text_projector(cond_text_states, t=timesteps, mask=cond_text_mask)
                if gen_token_indices is not None and self.training:
                    txt_hidden_states = txt_hidden_states.reshape(-1, txt_hidden_states.size(-1))[
                        cond_text_mask.view(-1).bool()].unsqueeze(0).contiguous()
            else:
                raise ValueError(f"text_proj_type {self._config.text_proj_type} not supported.")

            hidden_states, txt_hidden_states, audio_hidden_states = self.apply_padding(
                hidden_states, txt_hidden_states, audio_hidden_states, pad_count=pad_count
            )
            if audio_latents is not None:
                media_len = hidden_states.size(1) + audio_hidden_states.size(1)
            else:
                media_len = hidden_states.size(1)
            self.check_sizes(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                txt_hidden_states=txt_hidden_states,
                audio_hidden_states=audio_hidden_states,
                gen_token_indices=gen_token_indices,
                und_token_indices=und_token_indices,
                audio_token_indices=audio_token_indices,
                media_len=media_len,
            )

            seqlen = media_len + txt_hidden_states.size(1)
            device = hidden_states.device
            cos, sin = self.cached_rope(
                seqlen, device, rope_media_info=rope_media_info, sample_offsets=sample_offsets,
            )
            if get_parallel_state().cp_size > 1:
                hidden_states, txt_hidden_states = map(maybe_scatter_seq, [hidden_states, txt_hidden_states])
                if audio_latents is not None:
                    audio_hidden_states = maybe_scatter_seq(audio_hidden_states)

            if isinstance(timestep_states, list):
                timestep_states_tensor = torch.stack(timestep_states, dim=0)
            else:
                timestep_states_tensor = timestep_states
            if audio_latents is not None and isinstance(audio_timestep_states, list):
                audio_timestep_states_tensor = torch.stack(audio_timestep_states, dim=0)
            else:
                audio_timestep_states_tensor = audio_timestep_states

        else:
            # ----------- Visual -----------
            hidden_states_fl = torch.zeros(
                input_ids.size(0), input_ids.size(1), self._config.hidden_size,
                dtype=torch.bfloat16, device=input_ids.device,
            )

            if latents is not None:
                hidden_states_fl, timestep_states = self.instantiate_vae_media_tokens_full_seqlen(
                    self.patch_embed, self.time_embed, hidden_states_fl, timesteps, latents, visual_mask,
                    allowed_dims=[4, 5],
                )
                if isinstance(timestep_states, list):
                    timestep_states_tensor = torch.stack(timestep_states, dim=0)
                else:
                    timestep_states_tensor = timestep_states
            else:
                timestep_states = None
                timestep_states_tensor = None

            if timesteps_index is not None:
                hidden_states_fl = self.instantiate_continuous_tokens_full_seqlen(
                    hidden_states_fl, emb_layer=self.timestep_emb, scatter_src=timesteps, scatter_index=timesteps_index
                )

            # full sequence --> branch sequence (include pad/dummy tokens)
            gen_token_indices_ = gen_token_indices.unsqueeze(-1).expand(-1, -1, hidden_states_fl.shape[-1])
            hidden_states = hidden_states_fl.gather(dim=1, index=gen_token_indices_)

            # ----------- Text -----------
            txt_hidden_states_fl = torch.zeros(
                input_ids.size(0), input_ids.size(1), self._txt_config.hidden_size,
                dtype=torch.float32 if self._config.text_proj_type == "single_refiner" else torch.bfloat16,
                device=input_ids.device,
            )
            txt_hidden_states_fl = self.instantiate_text_tokens_full_seqlen(
                txt_hidden_states_fl, timesteps, cond_text_states, cond_text_mask, text_mask
            )

            und_token_indices_ = und_token_indices.unsqueeze(-1).expand(-1, -1, txt_hidden_states_fl.shape[-1])
            txt_hidden_states = txt_hidden_states_fl.gather(dim=1, index=und_token_indices_)

            # ----------- Audio -----------
            if audio_latents is not None:
                audio_hidden_states_fl = torch.zeros(
                    input_ids.size(0), input_ids.size(1), self._audio_config.hidden_size,
                    dtype=torch.bfloat16, device=input_ids.device,
                )
                audio_hidden_states_fl, audio_timestep_states = self.instantiate_vae_media_tokens_full_seqlen(
                    self.audio_projector, self.audio_time_embed,
                    audio_hidden_states_fl, audio_timesteps, audio_latents, audio_mask,
                    allowed_dims=[3, 4],
                )
                if isinstance(audio_timestep_states, list):
                    audio_timestep_states_tensor = torch.stack(audio_timestep_states, dim=0)
                else:
                    audio_timestep_states_tensor = audio_timestep_states

                audio_token_indices_ = audio_token_indices.unsqueeze(-1).expand(-1, -1, audio_hidden_states_fl.shape[-1])
                audio_hidden_states = audio_hidden_states_fl.gather(dim=1, index=audio_token_indices_)
            else:
                audio_timestep_states = None
                audio_hidden_states = None
                audio_timestep_states_tensor = None

            if get_parallel_state().cp_size > 1:
                hidden_states, txt_hidden_states = map(maybe_scatter_seq, [hidden_states, txt_hidden_states])
                if audio_latents is not None:
                    audio_hidden_states = maybe_scatter_seq(audio_hidden_states)

            seqlen = input_ids.size(1)
            device = hidden_states.device
            cos, sin = self.cached_rope(
                seqlen, device, rope_media_info=rope_media_info, sample_offsets=sample_offsets,
            )

        # ================== MeanFlow: build the r time-states ==================
        # Late layers + head do AdaLN with r; everything else (input tokens, text, the
        # first 3/4 layers) keeps using t. If r is not provided, reuse t (drop-in == LeoModel).
        n_t_layers = self._mf_num_t_layers()
        if latents is not None and timestep_r is not None:
            timestep_states_r = self._mf_time_states(self.time_embed, timestep_r)
            timestep_states_tensor_r = self._mf_stack(timestep_states_r)
        else:
            timestep_states_r = timestep_states if latents is not None else None
            timestep_states_tensor_r = timestep_states_tensor
        if audio_latents is not None and audio_timestep_r is not None:
            audio_timestep_states_r = self._mf_time_states(self.audio_time_embed, audio_timestep_r)
            audio_timestep_states_tensor_r = self._mf_stack(audio_timestep_states_r)
        elif audio_latents is not None:
            audio_timestep_states_r = audio_timestep_states
            audio_timestep_states_tensor_r = audio_timestep_states_tensor
        else:
            audio_timestep_states_r = None
            audio_timestep_states_tensor_r = None

        # Prepare transformer block inputs
        middle_layer_hidden_states = None
        # The anchor prediction u(t, t) is only needed for the training gradient forward.
        # NOTE: `_gan_disable_anchor` is an opt-in, default-OFF instance flag. It lets the
        # pure-GAN trainer (distill.gan_trainer) skip the anchor tail+head recompute
        # (u(t,t) is unused there — GAN only needs u(t,r)=diff_pred). It defaults to False,
        # so iMF / all existing training is completely unaffected.
        need_anchor = (
            self.training
            and (timestep_r is not None)
            and torch.is_grad_enabled()
            and not getattr(self, "_gan_disable_anchor", False)
        )

        # === Shared trunk: first n_t layers, AdaLN with t (common to u(t,r) and u(t,t)) ===
        for layer_idx in range(n_t_layers):
            layer = self.layers[layer_idx]
            if self._audio_config is not None:
                layer_inputs = [
                    (hidden_states, audio_hidden_states, txt_hidden_states),
                    (timestep_states_tensor, audio_timestep_states_tensor),
                    attention_mask,
                    (cos, sin),
                    (gen_token_indices, audio_token_indices, und_token_indices),
                    (gen_token_lengths, audio_token_lengths, und_token_lengths),
                ]
                hidden_states, audio_hidden_states, txt_hidden_states = layer(*layer_inputs)
            else:
                layer_inputs = [
                    (hidden_states, txt_hidden_states),
                    timestep_states_tensor,
                    attention_mask,
                    (cos, sin),
                    (gen_token_indices, und_token_indices),
                    (gen_token_lengths, und_token_lengths),
                ]
                hidden_states, txt_hidden_states = layer(*layer_inputs)

            # For repa (num_layers // 2 < n_t, so this is inside the shared trunk)
            if self.use_repa and self.training and layer_idx == self._config.num_layers // 2:
                if get_parallel_state().cp_size > 1:
                    hidden_states = maybe_gather_seq(hidden_states)
                    raise NotImplementedError("REPA + CP + SeqPack is not checked.")
                use_packing = layer_inputs[5] is not None and layer_inputs[5][0] is not None
                middle_layer_hidden_states = self._get_features_for_repa(hidden_states, rope_media_info, use_packing)
                if get_parallel_state().cp_size > 1:
                    hidden_states = maybe_scatter_seq(hidden_states)

        # === Tail layers + head, run from the shared trunk output ===
        # r-path -> u(t, r): last 1/4 layers + head do AdaLN with r. This is the main
        # prediction (used by the meanflow-consistency term).
        h_r, a_r = self._mf_run_tail(
            n_t_layers, hidden_states, audio_hidden_states, txt_hidden_states,
            timestep_states_tensor_r, audio_timestep_states_tensor_r, attention_mask, cos, sin,
            gen_token_indices, audio_token_indices, und_token_indices,
            gen_token_lengths, audio_token_lengths, und_token_lengths,
        )
        diff_pred, audio_diff_pred = self._mf_head(
            h_r, a_r, timestep_states_r, audio_timestep_states_r,
            input_ids, latents, audio_latents, gen_token_indices_, audio_token_indices_,
            visual_mask, audio_mask, rope_media_info, gen_token_lengths, audio_token_lengths, pad_count,
        )

        # anchor path -> u(t, t): last 1/4 layers + head do AdaLN with t. This is the
        # SEPARATE prediction the anchor term (pred(t,t) - v_tgt)^2 must use. Only the
        # last 1/4 layers + head are recomputed (the trunk is shared), so the extra cost
        # is small. Skipped outside the training gradient forward (and when r == t).
        if need_anchor:
            h_t, a_t = self._mf_run_tail(
                n_t_layers, hidden_states, audio_hidden_states, txt_hidden_states,
                timestep_states_tensor, audio_timestep_states_tensor, attention_mask, cos, sin,
                gen_token_indices, audio_token_indices, und_token_indices,
                gen_token_lengths, audio_token_lengths, und_token_lengths,
            )
            diff_pred_anchor, audio_diff_pred_anchor = self._mf_head(
                h_t, a_t, timestep_states, audio_timestep_states,
                input_ids, latents, audio_latents, gen_token_indices_, audio_token_indices_,
                visual_mask, audio_mask, rope_media_info, gen_token_lengths, audio_token_lengths, pad_count,
            )
        else:
            diff_pred_anchor = None
            audio_diff_pred_anchor = None

        # -- for inference
        if not self.training:
            if not return_dict:
                return diff_pred, audio_diff_pred
            return LeoOutput(
                diffusion_prediction=diff_pred,
                audio_diffusion_prediction=audio_diff_pred,
            )

        # === Calculate losses ===
        losses = {}
        global_metric_losses = {}
        loss = torch.tensor(0.0, device=hidden_states.device)
        use_global_diffusion_loss_average = getattr(self.args, "use_global_diffusion_loss_average", False)

        # -- diffusion loss
        if latents is not None:
            if diff_pred_anchor is not None:
                # iMF: consistency term uses pred(t, r); anchor term uses pred(t, t).
                raw_visual_loss = diffusion_loss_fn(
                    model_output=diff_pred, model_output_anchor=diff_pred_anchor
                )["loss"]
            else:
                raw_visual_loss = diffusion_loss_fn(model_output=diff_pred)["loss"]
            if use_global_diffusion_loss_average:
                if visual_loss_weight > 0:
                    diff_loss_sum, diff_loss_count = self._get_local_sum_and_count(
                        raw_visual_loss, diff_pred
                    )
                    losses["diff_loss_sum"] = diff_loss_sum
                    losses["diff_loss_count"] = diff_loss_count
                    losses["diff_loss_weight"] = visual_loss_weight
                    loss_key = f"{dataset_tag}_image_loss" if dataset_tag is not None else "image_loss"
                    global_metric_losses[loss_key] = (diff_loss_sum.detach().clone(), diff_loss_count.detach().clone())
                else:
                    visual_diff_loss = raw_visual_loss.mean()
                    loss = loss + visual_loss_weight * visual_diff_loss
            else:
                visual_diff_loss = raw_visual_loss.mean()
                if visual_loss_weight > 0:
                    loss_key = f"{dataset_tag}_image_loss" if dataset_tag is not None else "image_loss"
                    losses[loss_key] = visual_diff_loss.detach()
                loss = loss + visual_loss_weight * visual_diff_loss

        # -- audio diffusion loss
        if audio_latents is not None:
            if audio_diff_pred_anchor is not None:
                raw_audio_loss = audio_diffusion_loss_fn(
                    model_output=audio_diff_pred, model_output_anchor=audio_diff_pred_anchor
                )["loss"]
            else:
                raw_audio_loss = audio_diffusion_loss_fn(model_output=audio_diff_pred)["loss"]
            if use_global_diffusion_loss_average:
                if audio_loss_weight > 0:
                    audio_loss_sum, audio_loss_count = self._get_local_sum_and_count(
                        raw_audio_loss, audio_diff_pred
                    )
                    losses["audio_diff_loss_sum"] = audio_loss_sum
                    losses["audio_diff_loss_count"] = audio_loss_count
                    losses["audio_diff_loss_weight"] = audio_loss_weight
                    loss_key = f"{dataset_tag}_audio_loss" if dataset_tag is not None else "audio_loss"
                    global_metric_losses[loss_key] = (audio_loss_sum.detach().clone(), audio_loss_count.detach().clone())
                else:
                    audio_diff_loss = raw_audio_loss.mean()
                    loss = loss + audio_loss_weight * audio_diff_loss
            else:
                audio_diff_loss = raw_audio_loss.mean()
                if audio_loss_weight > 0:
                    loss_key = f"{dataset_tag}_audio_loss" if dataset_tag is not None else "audio_loss"
                    losses[loss_key] = audio_diff_loss.detach()
                loss = loss + audio_loss_weight * audio_diff_loss

        if global_metric_losses:
            losses["_global_metric_losses"] = global_metric_losses

        loss, losses = self.get_aux_losses(
            loss,
            losses,
            repa_feats=repa_feats,
            middle_layer_hidden_states=middle_layer_hidden_states,
            use_global_diffusion_loss_average=use_global_diffusion_loss_average,
        )

        # -- total loss (for backward)
        losses["loss"] = loss

        if not return_dict:
            return losses, diff_pred

        return LeoOutput(
            losses=losses,
            diffusion_prediction=diff_pred,
            audio_diffusion_prediction=audio_diff_pred,
        )


# --------------------------------------------------------------------------- #
# Non-invasive dispatch extension for `model-structure: LeoModelMeanFlow`.
# --------------------------------------------------------------------------- #
def _build_meanflow(args, **kw):
    from .leo_config import LeoConfig, core_model_config_from_args

    device = kw.get("device", None)
    dtype = kw.get("dtype", None)
    factor_kwargs = {"device": device, "dtype": dtype}

    model_name = args.model_name.split(".")[-1]
    model_config_dict = core_model_config_from_args(args)
    model_config = LeoConfig.from_name(model_name, **model_config_dict)

    if args.text_branch_model_name is not None:
        text_branch_config_dict = core_model_config_from_args(args, prefix="text_branch_")
        text_branch_model_name = args.text_branch_model_name.split(".")[-1]
        text_branch_config = LeoConfig.from_name(text_branch_model_name, **text_branch_config_dict)
    else:
        text_branch_config = None

    if args.audio_branch_model_name is not None:
        audio_branch_config_dict = core_model_config_from_args(args, prefix="audio_branch_")
        audio_branch_model_name = args.audio_branch_model_name.split(".")[-1]
        audio_branch_config = LeoConfig.from_name(audio_branch_model_name, **audio_branch_config_dict)
    else:
        audio_branch_config = None

    model = LeoModelMeanFlow(
        args, model_config, txt_config=text_branch_config, audio_config=audio_branch_config,
        **factor_kwargs,
    )
    return model, dict(
        main_branch=model_config, text_branch=text_branch_config, audio_branch=audio_branch_config,
    )


def install_meanflow_dispatch():
    import hymm.models.diffusion as _pkg

    current = getattr(_pkg, "build_model", None)
    if current is None or getattr(current, "_meanflow_patched", False):
        return

    real_build = current

    def patched_build_model(args, *a, **kw):
        if getattr(args, "model_structure", None) == "LeoModelMeanFlow":
            return _build_meanflow(args, *a, **kw)
        return real_build(args, *a, **kw)

    patched_build_model._meanflow_patched = True
    patched_build_model._orig_build_model = real_build
    _pkg.build_model = patched_build_model


# Auto-install on import.
install_meanflow_dispatch()
