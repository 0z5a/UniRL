from copy import deepcopy
from collections import defaultdict
from hymm.data_kits.caption_strategy import load_caption_processor
import json
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import os
from hymm.core.extra_model_provider import build_denoiser


def build_av_denoiser(args):
    # Modality-specific SNR config; fall back to global args when not set (None).
    video_snr_type = args.flow_snr_type_video or args.flow_snr_type
    video_snr_mix_ratio = (
        args.flow_snr_mix_uniform_ratio_video
        if args.flow_snr_mix_uniform_ratio_video is not None
        else args.flow_snr_mix_uniform_ratio
    )
    video_denoiser = build_denoiser(
        denoiser_type="video",
        shift=args.flow_shift_video,
        snr_type=video_snr_type,
        snr_mix_uniform_ratio=video_snr_mix_ratio,
    )
    audio_denoiser = None
    if args.audio_branch_model_name is not None:
        audio_snr_type = args.flow_snr_type_audio or args.flow_snr_type
        audio_snr_mix_ratio = (
            args.flow_snr_mix_uniform_ratio_audio
            if args.flow_snr_mix_uniform_ratio_audio is not None
            else args.flow_snr_mix_uniform_ratio
        )
        audio_denoiser = build_denoiser(
            denoiser_type="audio",
            shift=args.flow_shift_audio,
            snr_type=audio_snr_type,
            snr_mix_uniform_ratio=audio_snr_mix_ratio,
        )
    return video_denoiser, audio_denoiser

def _resolve_path(path: str, cos_base: str = "") -> Path:
    path = str(path).strip()
    if not path.endswith(".npy"):
        path = f"{path}.npy"
    if not path.startswith(cos_base) and not os.path.isabs(path):
        path = cos_base + "/" + path
    return Path(path)


def load_video_latent(path: str, latent_dim: int, cos_base: str = "") -> torch.Tensor:
    """Load video latent as ``[C, T, H, W]`` float32."""

    latent = np.load(_resolve_path(path, cos_base))
    assert latent.ndim == 5 and latent.shape[0] == 1 and latent.shape[1] == latent_dim, (
        f"Video latent should be [B, C, T, H, W] with B=1, C={latent_dim}, got {latent.shape}"
    )
    return torch.from_numpy(latent.astype(np.float32))


def load_audio_latent(path: str, latent_dim: int, cos_base: str = "") -> torch.Tensor:
    """Load audio latent as ``[C, L]`` float32."""

    latent = np.load(_resolve_path(path, cos_base))
    latent = torch.from_numpy(latent.astype(np.float32))
    assert latent.dim() == 3 and latent.size(1) == latent_dim, (
        f"Audio latent should be [B, C, L] with C={latent_dim}, got {tuple(latent.shape)}"
    )
    return latent

def latent_shape_to_media_size(
    latent_shape: tuple[int, int, int, int, int],
    d_factor: int,
    h_factor: int,
    w_factor: int,
) -> tuple[int, int, int]:
    """Return ``(num_frames, height, width)`` from ``(1, C, T, H, W)`` latent shape."""
    _, _, tk_d, tk_h, tk_w = latent_shape
    num_frames = (tk_d - 1) * d_factor + 1
    height = tk_h * h_factor
    width = tk_w * w_factor
    return num_frames, height, width

def parse_latent_shape(raw_shape) -> tuple[int, int, int, int, int]:
    if isinstance(raw_shape, str):
        raw_shape = json.loads(raw_shape)
    if isinstance(raw_shape[0], list):
        raw_shape = raw_shape[0]
    assert len(raw_shape) == 5, f"latent_shape must have 5 dims, got {raw_shape}"
    return tuple(int(x) for x in raw_shape)

def build_validation_caption_processor(args, logger=None):
    """Build the training ``video_caption_v1`` CaptionAug from t2va_task_kwargs.

    Mirrors ``MultiCaptionManager`` setup so validation-loss prompts go through the
    SAME structured-caption assembly (``caption_sample_ratio`` /
    ``caption_processor_kwargs``) as training. The en/zh ``*_src`` selection is
    intentionally ignored -- the caption column is pinned explicitly per spec via
    ``@@prompt_<label>=<col>``.

    Returns ``None`` if no caption processor is configured (falls back to the raw
    column text).
    """
    task_kwargs = deepcopy(getattr(args, "t2va_task_kwargs", {}))
    resource = (task_kwargs or {}).get("video_caption_resource")
    if not resource:
        return None
    for item in resource:
        if not isinstance(item, dict) or not item:
            continue
        group_key = next(iter(item.keys()))
        raw = item[group_key] or {}
        prefix = f"{group_key}_"
        stripped = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in raw.items()
        }
        if "caption_processor" not in stripped:
            continue
        return load_caption_processor(
            name=stripped["caption_processor"],
            caption_sample_ratio=stripped.get("caption_sample_ratio"),
            logger=logger,
            kwargs=stripped.get("caption_processor_kwargs"),
        )
    return None


# ===========================================================================
# Validation-loss set parsing / storage / aggregation / plotting
# ===========================================================================

def parse_iter_from_name(name) -> int:
    """Extract the training iteration from a sample-save dir name like ``iter_0000005``."""
    m = re.search(r"iter_(\d+)", str(name))
    return int(m.group(1)) if m else 0


def sanitize_key(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.=-]+", "_", str(s))


def parse_validation_loss_set(spec: str) -> list[dict]:
    """Parse one ``VALIDATION_LOSS_SETS`` entry into per-language variants.

    Format (shares the ``@@key=value`` convention with ``--testsets``)::

        <csv>@@plot_name=<name>@@prompt_zh=<col>@@prompt_en=<col>[@@_first_n_=N|@@_last_n_=N]
        <csv>@@plot_name=<name>@@prompt=<col>[@@_first_n_=N|@@_last_n_=N]

    Returns a list of variant dicts, each with:
        ``plot_name``, ``csv_stem``, ``lang`` ("zh"/"en"), ``label`` (legend label),
        ``prompt_col``, ``testset`` (a per-variant string reusing ``MessageListDataset``
        via ``@@prompt=<col>`` plus the subset kwarg), and optionally ``warn_default_lang``.
    """
    parts = spec.split("@@")
    csv_path = parts[0]
    kw = {}
    for part in parts[1:]:
        if not part:
            continue
        key, value = part.split("=", 1)
        kw[key] = value

    plot_name = kw.get("plot_name", Path(csv_path).stem)
    csv_stem = Path(csv_path).stem

    # Column names (in the CSV) that hold the video / audio latent paths. Configurable via
    # @@video_latent_path=<col>@@audio_latent_path=<col>; default to the common names.
    video_latent_col = kw.get("video_latent_path", "latent_cos_path")
    audio_latent_col = kw.get("audio_latent_path", "audio_av_clip_vae_leo_v1_0_0_latent_cos_path")

    # Preserve the subset selector (first/last N) so each variant reads the same rows.
    subset_parts = []
    if "_first_n_" in kw:
        subset_parts.append(f"_first_n_={kw['_first_n_']}")
    if "_last_n_" in kw:
        subset_parts.append(f"_last_n_={kw['_last_n_']}")

    variants = []
    if "prompt_zh" in kw:
        variants.append(dict(lang="zh", label="zh", prompt_col=kw["prompt_zh"]))
    if "prompt_en" in kw:
        variants.append(dict(lang="en", label="en", prompt_col=kw["prompt_en"]))
    if not variants and "prompt" in kw:
        # Bare ``prompt``: default caption language to "zh" (with a warning) and use the
        # column name itself as the legend label.
        variants.append(dict(lang="zh", label=kw["prompt"], prompt_col=kw["prompt"],
                             warn_default_lang=True))
    if not variants:
        raise ValueError(
            f"Validation-loss set must specify prompt_zh/prompt_en/prompt: {spec}"
        )

    for v in variants:
        testset = csv_path + f"@@prompt={v['prompt_col']}"
        for sp in subset_parts:
            testset += f"@@{sp}"
        v["testset"] = testset
        v["plot_name"] = plot_name
        v["csv_stem"] = csv_stem
        v["video_latent_col"] = video_latent_col
        v["audio_latent_col"] = audio_latent_col
    return variants


def variant_cache_dir(val_dir, iter_num: int, variant: dict) -> Path:
    key = sanitize_key(f"{variant['plot_name']}__{variant['csv_stem']}__{variant['label']}")
    return Path(val_dir) / "cache" / f"iter_{iter_num:07d}" / key


def load_done_indices(cache_file, n_ts: int) -> set:
    """Return the set of sample indices already fully computed (all timesteps present)."""
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return set()
    try:
        df = pd.read_csv(cache_file)
    except Exception:
        return set()
    if "index" not in df.columns or "timestep_idx" not in df.columns:
        return set()
    df["index"] = df["index"].astype(str)
    counts = df.groupby("index")["timestep_idx"].nunique()
    return set(str(i) for i in counts[counts >= n_ts].index.tolist())


def append_sample_rows(cache_file, index: int, outputs: dict) -> None:
    """Append one row per timestep for a single sample (crash-safe incremental write)."""
    cache_file = Path(cache_file)
    ts = outputs["timesteps"]
    vl = outputs["video_losses"]
    al = outputs["audio_losses"]
    rows = []
    for k in range(len(ts)):
        rows.append(dict(
            index=index,
            timestep_idx=k,
            timestep=ts[k],
            video_loss="" if vl[k] is None else vl[k],
            audio_loss="" if al[k] is None else al[k],
        ))
    df = pd.DataFrame(rows)
    df.to_csv(cache_file, mode="a", header=not cache_file.exists(), index=False)


def shift_weighted_mean(per_timestep, timestep_points, shift):
    """Weight per-timestep losses by the timestep density that the flow shift induces.

    Validation loss is measured on a UNIFORM grid of ``t``, but training/inference do not visit
    ``t`` uniformly: a base ``t`` is mapped through the constant (SD3) shift
    ``t -> shift * t / (1 + (shift - 1) * t)`` (``Transport._apply_shift`` with ``flow-reverse:
    True``, and ``FlowMatchDiscreteScheduler.sd3_time_shift`` at inference). Pushing a uniform
    base through that map gives the density

        p(t) = shift / (shift - (shift - 1) * t) ** 2      (integrates to 1 over [0, 1])

    so re-weighting the uniform grid by ``p(t)`` turns the plain mean into an estimate of the loss
    over the timesteps actually sampled with that shift. ``shift == 1`` reproduces the plain mean.

    Note this models the shift only; a non-uniform base distribution (``flow-snr-type lognorm`` /
    ``uniform_lognorm_mix``) is not folded in. Returns ``None`` when the shift is unknown or no
    timestep has a value.
    """
    if not per_timestep or shift is None:
        return None
    shift = float(shift)
    if shift <= 0:
        return None
    weights, values = [], []
    for t, v in zip(timestep_points, per_timestep):
        if v is None:
            continue
        denom = shift - (shift - 1.0) * float(t)
        if denom <= 0:
            continue
        weights.append(shift / denom ** 2)
        values.append(float(v))
    if not values:
        return None
    return float(np.average(values, weights=weights))


def aggregate_validation_iter(val_dir, iter_num: int, timestep_points, variants: list[dict],
                              flow_shifts: Optional[dict] = None) -> dict:
    """Aggregate per-sample cache files (all ranks) into per-variant per-timestep means.

    The average is taken over all unique samples (dedup by (index, timestep_idx) across ranks),
    which matches the training ``Loss/*_image_loss`` / ``Loss/*_audio_loss`` (mean of per-sample
    ``mean_flat`` velocity MSE over the dataset).

    ``flow_shifts`` is ``{"video": <flow_shift_video>, "audio": <flow_shift_audio>}``; it is stored
    in the results json and used for the ``shift_weighted_mean`` of each modality.
    """
    cache_root = Path(val_dir) / "cache" / f"iter_{iter_num:07d}"
    n_ts = len(timestep_points)
    flow_shifts = flow_shifts or {}
    results = {
        "iter": int(iter_num),
        "timesteps": [float(t) for t in timestep_points],
        "flow_shifts": {
            m: (None if flow_shifts.get(m) is None else float(flow_shifts[m]))
            for m in ("video", "audio")
        },
        "variants": [],
    }

    for v in variants:
        cdir = variant_cache_dir(val_dir, iter_num, v)
        if not cdir.exists():
            continue
        dfs = []
        for f in sorted(cdir.glob("rank_*.csv")):
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                continue
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        df["index"] = df["index"].astype(str)
        df = df.drop_duplicates(subset=["index", "timestep_idx"], keep="last")

        def per_ts(col):
            vals = []
            for k in range(n_ts):
                sub = pd.to_numeric(df[df["timestep_idx"] == k][col], errors="coerce").dropna()
                vals.append(float(sub.mean()) if len(sub) else None)
            valid = [x for x in vals if x is not None]
            return vals, (float(np.mean(valid)) if valid else None)

        vpt, vmean = per_ts("video_loss")
        apt, amean = per_ts("audio_loss")
        results["variants"].append(dict(
            plot_name=v["plot_name"], csv_stem=v["csv_stem"], label=v["label"],
            video=dict(per_timestep=vpt, mean=vmean,
                       shift_weighted_mean=shift_weighted_mean(
                           vpt, timestep_points, results["flow_shifts"]["video"])),
            audio=dict(per_timestep=apt, mean=amean,
                       shift_weighted_mean=shift_weighted_mean(
                           apt, timestep_points, results["flow_shifts"]["audio"])),
        ))

    out_dir = Path(val_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"iter_{iter_num:07d}.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def _avg_ignore_none(values):
    valid = [x for x in values if x is not None]
    return float(np.mean(valid)) if valid else None


def plot_validation_losses(val_dir) -> None:
    """Read every ``results/iter_*.json`` and render one PNG per (plot_name, modality).

    Each figure has ``n_timesteps + 2`` subplots, laid out 4 per row: one per timestep, then the
    plain all-timestep mean, then the shift-weighted mean (see ``shift_weighted_mean``).
    X axis is the checkpoint iteration; each line is a caption-language variant (``zh``/``en`` or
    the raw column name). When a plot group has multiple CSVs, each point averages those CSVs.
    """
    # Point matplotlib's config/cache at a guaranteed-writable dir before importing it. On clusters
    # $HOME is often read-only, which can make the first-time font-cache build error out or stall.
    mpl_cache = Path(val_dir) / ".mplcache"
    try:
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    except Exception:
        pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(val_dir) / "results"
    files = sorted(results_dir.glob("iter_*.json"))
    if not files:
        return

    iters_data = []
    for f in files:
        try:
            with open(f) as fh:
                iters_data.append(json.load(fh))
        except Exception:
            continue
    iters_data.sort(key=lambda d: d["iter"])
    if not iters_data:
        return

    # accum[plot_name][modality][label][iter] -> list of variant-modality dicts (one per CSV)
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    timesteps_by_plot = {}
    # [plot_name][modality] -> flow shift used, for the shift-weighted panel title (latest iter wins)
    shift_by_plot = defaultdict(dict)
    for d in iters_data:
        it = int(d["iter"])
        flow_shifts = d.get("flow_shifts") or {}
        for var in d["variants"]:
            pn = var["plot_name"]
            timesteps_by_plot[pn] = d["timesteps"]
            for modality in ("video", "audio"):
                m = var[modality]
                shift = flow_shifts.get(modality)
                # Results written before the shift-weighted panel existed have no
                # ``shift_weighted_mean``; recompute it here when the shift is known.
                if m.get("shift_weighted_mean") is None:
                    m["shift_weighted_mean"] = shift_weighted_mean(
                        m.get("per_timestep"), d["timesteps"], shift)
                if shift is not None:
                    shift_by_plot[pn][modality] = float(shift)
                accum[pn][modality][var["label"]][it].append(m)

    plots_dir = Path(val_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for pn, per_modality in accum.items():
        ts = timesteps_by_plot[pn]
        n_ts = len(ts)
        # per-timestep panels + plain mean + shift-weighted mean
        n_panels = n_ts + 2
        for modality, per_label in per_modality.items():
            # Skip a modality with no numeric data (e.g. audio missing for a video-only run).
            has_data = any(
                _avg_ignore_none([m["mean"] for m in per_iter])
                is not None
                for per_label_iters in per_label.values()
                for per_iter in per_label_iters.values()
            )
            if not has_data:
                continue

            ncols = min(4, n_panels)
            nrows = int(np.ceil(n_panels / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
            axes_flat = axes.flatten()

            all_iters = sorted({it for per_iter in per_label.values() for it in per_iter})
            if all_iters:
                lo, hi = min(all_iters), max(all_iters)
                xlim = (lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)
            else:
                xlim = (0, 1)

            all_yvals = []  # collect every plotted value to share one y-range across subplots
            for label, per_iter in sorted(per_label.items()):
                iters_sorted = sorted(per_iter.keys())
                # Per-timestep panels
                for k in range(n_ts):
                    ys = [_avg_ignore_none([m["per_timestep"][k] for m in per_iter[it]])
                          for it in iters_sorted]
                    xs = [it for it, y in zip(iters_sorted, ys) if y is not None]
                    yv = [y for y in ys if y is not None]
                    if xs:
                        axes_flat[k].plot(xs, yv, marker="o", label=label)
                        all_yvals.extend(yv)
                # Mean panel
                ys = [_avg_ignore_none([m["mean"] for m in per_iter[it]]) for it in iters_sorted]
                xs = [it for it, y in zip(iters_sorted, ys) if y is not None]
                yv = [y for y in ys if y is not None]
                if xs:
                    axes_flat[n_ts].plot(xs, yv, marker="o", label=label)
                    all_yvals.extend(yv)
                # Shift-weighted mean panel
                ys = [_avg_ignore_none([m.get("shift_weighted_mean") for m in per_iter[it]])
                      for it in iters_sorted]
                xs = [it for it, y in zip(iters_sorted, ys) if y is not None]
                yv = [y for y in ys if y is not None]
                if xs:
                    axes_flat[n_ts + 1].plot(xs, yv, marker="o", label=label)
                    all_yvals.extend(yv)

            # Shared y-range across all subplots (for the GlobalY_ variant); LocalY_ keeps each subplot's
            # own autoscaled range.
            if all_yvals:
                ymin, ymax = min(all_yvals), max(all_yvals)
                pad = (ymax - ymin) * 0.05 if ymax > ymin else (abs(ymax) * 0.05 or 1.0)
                shared_ylim = (ymin - pad, ymax + pad)
            else:
                shared_ylim = None

            shift = shift_by_plot.get(pn, {}).get(modality)
            shift_title = ("shift-weighted mean (shift=n/a)" if shift is None
                           else f"shift-weighted mean (shift={shift:g})")
            titles = ([f"t={ts[k]:.3f}" for k in range(n_ts)]
                      + ["mean (all timesteps)", shift_title])
            for k in range(n_panels):
                axes_flat[k].set_title(titles[k])
                axes_flat[k].set_xlabel("iter")
                axes_flat[k].set_ylabel("loss")
                axes_flat[k].set_xlim(*xlim)
                axes_flat[k].grid(True, alpha=0.3)
            # Hide any unused panels.
            for k in range(n_panels, len(axes_flat)):
                axes_flat[k].axis("off")

            # Collect a de-duplicated legend across all panels and place it at the very top.
            legend_map = {}
            for ax in axes_flat[:n_panels]:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    legend_map.setdefault(label, handle)
            fig.suptitle(f"{pn} - {modality} validation loss", y=0.995)
            if legend_map:
                # Legend just below the title, above all subplots (never inside a panel).
                fig.legend(list(legend_map.values()), list(legend_map.keys()),
                           loc="upper center", bbox_to_anchor=(0.5, 0.955),
                           ncol=max(1, len(legend_map)), frameon=False)
            # Reserve headroom at the top for the suptitle + legend.
            fig.tight_layout(rect=(0, 0, 1, 0.9))

            # LocalY_: each subplot keeps its own auto-scaled y-range (drawn above by default).
            fig.savefig(plots_dir / f"LocalY_{pn}_{modality}.png", dpi=120)

            # GlobalY_: force the same y-range on every subplot for direct cross-timestep comparison.
            if shared_ylim is not None:
                for k in range(n_panels):
                    axes_flat[k].set_ylim(*shared_ylim)
            fig.savefig(plots_dir / f"GlobalY_{pn}_{modality}.png", dpi=120)

            plt.close(fig)