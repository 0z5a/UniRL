# improved-MeanFlow (iMF) distillation loss helpers.
#
# This is a NEW, self-contained module. It does NOT modify any existing file.
# It provides the ragged-aware math used by `hymm.trainers.distill.imf_trainer.IMFTrainer`
# to build a per-branch iMF loss that is injected into the model's existing
# `diffusion_loss_fn` / `audio_diffusion_loss_fn` hook (see `leo.py:LeoModel.forward`).
#
# ============================ Background ============================
# iMF / MeanFlow distillation learns the *average* velocity u(z, r, t) over [r, t]:
#       u(z, r, t) = v(z, t) - (t - r) * du/dt
# CFG self-distillation anchor:
#       v_tgt = u_t + (1 - 1/w) * (v_cond - v_uncond)
# du/dt via central finite difference along the ODE trajectory.
# Per-branch loss (under the no-timestep_r backbone, pred(t,r) == pred(t,t)):
#       L = mean( (pred - v_tgt + (t - r) * du/dt) ** 2 )
#         + anchor_weight * mean( (pred - v_tgt) ** 2 )
#
# ============================ Ragged structure (IMPORTANT) ============================
# In packed training the fields do NOT all share the same nesting:
#   * data fields produced by the VAE pipeline (`latents` = x_t, `u_t`) are list[B]
#     whose per-item entry is a STACKED tensor [N_i, C, ...] when the N_i sub-samples
#     share a shape, or a list[N_i] of tensors when they differ.
#   * the model PREDICTION (diff_pred / audio_diff_pred) is list[B] of list[N_i] of
#     tensors [1, C, ...] (one level deeper).
#   * time fields (t, r, model_t) are list[B] of 1-D tensors [N_i].
# `Transport.ragged_mse` reconciles the prediction (list[B] of list) against the
# target (list[B] of stacked tensor) by zipping a list against a tensor, i.e. iterating
# the stacked tensor along dim 0. The walkers below do exactly that: they descend
# whenever ANY field at the current level is a list, indexing the other (tensor) fields
# along dim 0, and they preserve the *driver* field's structure (stacking leaf results
# back into a tensor when the driver was a tensor that we had to iterate).

import torch as th

from .utils import mean_flat


def _is_seq(x):
    return isinstance(x, (list, tuple))


# --------------------------------------------------------------------------- #
# ndim alignment for data leaves (a pred leaf may carry an extra leading 1).
# --------------------------------------------------------------------------- #
def _match_ndim(x, ref):
    while x.ndim > ref.ndim and x.shape[0] == 1:
        x = x.squeeze(0)
    while x.ndim < ref.ndim:
        x = x.unsqueeze(0)
    assert x.ndim == ref.ndim, (
        f"Cannot align ndim: x{tuple(x.shape)} vs ref{tuple(ref.shape)}"
    )
    return x


# --------------------------------------------------------------------------- #
# indexing / lengths that work for both lists and stacked tensors.
# --------------------------------------------------------------------------- #
def _seq_len(fields):
    for f in fields:
        if _is_seq(f):
            return len(f)
    for f in fields:
        if isinstance(f, th.Tensor):
            return f.shape[0]
    raise ValueError("No sequence-like field to determine length from.")


def _get(f, k):
    if _is_seq(f):
        return f[k]
    if isinstance(f, th.Tensor):
        return f[k]
    return f  # scalar / None: shared across sub-items


def _index_time(time, k):
    if _is_seq(time):
        return time[k]
    if isinstance(time, th.Tensor) and time.ndim >= 1:
        return time[k]
    return time  # scalar shared


def _broadcast_time(time, ref_leaf):
    """Reshape a per-sub-sample time value to broadcast against ``ref_leaf``."""
    if not isinstance(time, th.Tensor):
        return time
    t = time.float()
    if t.ndim == 0:
        return t
    if ref_leaf.ndim >= 1 and t.shape[0] == ref_leaf.shape[0]:
        return t.view(t.shape[0], *([1] * (ref_leaf.ndim - 1)))
    if t.numel() == 1:
        return t.reshape(())
    return t.view(t.shape[0], *([1] * max(ref_leaf.ndim - 1, 0)))


def _should_descend(fields):
    return any(_is_seq(f) for f in fields)


# --------------------------------------------------------------------------- #
# generic recursion walkers (driver == fields[0]; structure-preserving).
# --------------------------------------------------------------------------- #
def _walk_fields(fields, leaf_fn):
    if _should_descend(fields):
        n = _seq_len(fields)
        results = [_walk_fields(tuple(_get(f, k) for f in fields), leaf_fn) for k in range(n)]
        if isinstance(fields[0], th.Tensor):
            return th.stack(results, dim=0)
        return results
    return leaf_fn(*fields)


def _walk_fields_time(fields, time, leaf_fn):
    if _should_descend(fields):
        n = _seq_len(fields)
        results = [
            _walk_fields_time(tuple(_get(f, k) for f in fields), _index_time(time, k), leaf_fn)
            for k in range(n)
        ]
        if isinstance(fields[0], th.Tensor):
            return th.stack(results, dim=0)
        return results
    return leaf_fn(*fields, _broadcast_time(time, fields[0]))


# --------------------------------------------------------------------------- #
# pure-time helpers (preserve the time structure; leaves are 1-D / scalar).
# --------------------------------------------------------------------------- #
def time_unary(t, fn):
    if _is_seq(t):
        return [time_unary(x, fn) for x in t]
    return fn(t)


def time_binary(a, b, fn):
    if _is_seq(a):
        return [time_binary(x, y, fn) for x, y in zip(a, b)]
    return fn(a, b)


def rand_like_scaled(t, generator=None):
    """r = U(0, t) per time leaf, preserving the time structure."""
    if _is_seq(t):
        return [rand_like_scaled(x, generator=generator) for x in t]
    u = th.empty_like(t, dtype=th.float32).uniform_(0.0, 1.0, generator=generator)
    return (u * t.float()).to(t.dtype)


def broadcast_time_(t, src, group):
    """In-place dist.broadcast of every time leaf (for CP-group consistency)."""
    import torch.distributed as dist
    if _is_seq(t):
        for x in t:
            broadcast_time_(x, src, group)
    else:
        dist.broadcast(t, src=src, group=group)


# --------------------------------------------------------------------------- #
# v_tgt = u_t + (1 - 1/w) * (v_cond - v_uncond)  (driver = v_cond -> pred structure)
# --------------------------------------------------------------------------- #
def compute_cfg_target(ut, v_cond, v_uncond, guidance):
    coef = 1.0 - 1.0 / float(guidance)

    def leaf(c, n, u):
        c = c.float()
        n = n.float()
        u = _match_ndim(u.float(), c)
        return u + coef * (c - n)

    return _walk_fields((v_cond, v_uncond, ut), leaf)


# --------------------------------------------------------------------------- #
# du/dt central finite difference (driver = v_plus -> pred structure)
# --------------------------------------------------------------------------- #
def compute_dudt(v_plus, v_minus, span):
    def leaf(vp, vm, span_b):
        return (vp.float() - vm.float()) / span_b

    return _walk_fields_time((v_plus, v_minus), span, leaf)


# --------------------------------------------------------------------------- #
# trajectory perturbation: x' = x + step * v (channel-sliced; driver = latents)
# --------------------------------------------------------------------------- #
def perturb_latents(latents, velocity, step):
    def leaf(lat, vel, step_b):
        return _perturb_leaf(lat, vel, step_b)

    return _walk_fields_time((latents, velocity), step, leaf)


def _perturb_leaf(lat, vel, step_b):
    vel = _match_ndim(vel, lat)
    delta = step_b * vel.float()
    diff_dims = [d for d in range(lat.ndim) if lat.shape[d] != vel.shape[d]]
    out = lat.clone()
    if len(diff_dims) == 0:
        return out + delta.to(out.dtype)
    assert len(diff_dims) == 1, (
        f"Ambiguous channel dim between latent{tuple(lat.shape)} and "
        f"velocity{tuple(vel.shape)}; expected exactly one differing dim."
    )
    cd = diff_dims[0]
    cv = vel.shape[cd]
    assert lat.shape[cd] >= cv, (
        f"latent channels ({lat.shape[cd]}) < velocity channels ({cv}) at dim {cd}"
    )
    idx = [slice(None)] * lat.ndim
    idx[cd] = slice(0, cv)
    idx = tuple(idx)
    out[idx] = out[idx] + delta.to(out.dtype)
    return out


# --------------------------------------------------------------------------- #
# per-batch-item iMF loss (mirrors Transport.ragged_mse reduction -> [B])
#
# Two DIFFERENT predictions are used (matching the reference
# `training_losses_t2va_imf_v2`):
#   * pred        = model(t, timestep_r=r)  -> meanflow-consistency term
#                       ( pred - v_tgt + (t - r) * du/dt )^2
#   * pred_anchor = model(t, timestep_r=t)  -> anchor term
#                       anchor_weight * ( pred_anchor - v_tgt )^2
# When pred_anchor is None (e.g. r == t / non-meanflow backbone) it falls back to
# pred, recovering the old single-prediction behaviour.
# --------------------------------------------------------------------------- #
def imf_loss_per_item(pred, pred_anchor, v_tgt, dudt, time_diff, anchor_weight):
    if pred_anchor is None:
        pred_anchor = pred

    if isinstance(pred, th.Tensor):
        tb = _broadcast_time(time_diff, pred)
        return _imf_meanflat(pred, pred_anchor, v_tgt, dudt, tb, anchor_weight)

    batch = []
    for i in range(len(pred)):
        batch.append(
            _imf_item(
                _get(pred, i), _get(pred_anchor, i), _get(v_tgt, i), _get(dudt, i),
                _index_time(time_diff, i), anchor_weight,
            )
        )
    return th.cat(batch)


def _imf_item(p, pa, vt, du, td, aw):
    if isinstance(p, th.Tensor):
        tb = _broadcast_time(td, p)
        return _imf_mean(p, pa, vt, du, tb, aw)[None]
    subs = []
    for k in range(len(p)):
        pk = _get(p, k)
        tb = _broadcast_time(_index_time(td, k), pk)
        subs.append(_imf_mean(pk, _get(pa, k), _get(vt, k), _get(du, k), tb, aw)[None])
    return sum(subs) / len(subs)


def _imf_mean(p, pa, vt, du, tb, aw):
    vt = _match_ndim(vt, p)
    du = _match_ndim(du, p)
    pa = _match_ndim(pa, p)
    flow = p.float() - vt + tb * du
    anchor = pa.float() - vt
    return 0.1 * flow.pow(2).mean() + 1 * anchor.pow(2).mean()


def _imf_meanflat(p, pa, vt, du, tb, aw):
    vt = _match_ndim(vt, p)
    du = _match_ndim(du, p)
    pa = _match_ndim(pa, p)
    flow = p.float() - vt + tb * du
    anchor = pa.float() - vt
    return 0.1 * mean_flat(flow.pow(2)) + 1 * mean_flat(anchor.pow(2))


# --------------------------------------------------------------------------- #
# loss-fn factory: drop-in replacement for `*_diffusion_loss_fn`
# --------------------------------------------------------------------------- #
def make_imf_loss_fn(time_diff, v_tgt, dudt, anchor_weight):
    def _loss_fn(model_output, model_output_anchor=None, **_unused):
        return {
            "loss": imf_loss_per_item(
                model_output, model_output_anchor, v_tgt, dudt, time_diff, anchor_weight
            )
        }

    return _loss_fn
