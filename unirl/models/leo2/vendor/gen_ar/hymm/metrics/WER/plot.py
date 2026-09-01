"""Plot training-step vs WER curves for ASR evaluation runs.

Each variant in the module-level `info` dict maps to a list of *root* dirs.
Under each root, sub-folders named `iter_<step>` contain a `wer_results.json`
whose `overall_wer` field is the y-value and `num_samples` is shown in the
plot title. If multiple roots are given, their `iter_<step>` entries are
merged (the user has confirmed iters never collide across the list).

Experiments to plot are listed in `exps`: each entry picks one or more
variants from `info` to draw on a shared figure, saved to `output`. An exp
may also carry an optional `index_filter` (list of stringified sample
indices) to restrict the y-value to a subset of `details` — see the
`Index filtering` section below.

Style is adapted from `vis/plots_CLIP_FID_HPSv2_metrics_utils.py` so the
WER curves look consistent with the FID / CLIP / HPSv2 curves.

Index filtering (alternative aggregation):
    By default the curve uses `overall_wer` straight from the JSON. If a
    subset of sample indices is supplied (e.g. ['9', '16']), the y-value at
    each step becomes the simple mean of per-sample `wer` over `details`
    whose `index` matches one of the supplied values. Indices in the filter
    that aren't present at a step are reported in WARN logs; steps where
    *no* index matches are silently skipped on that curve.

    Two ways to supply the filter:
      1) Per-exp in code:
             exps["my_exp"]["index_filter"] = ["9", "16", "37"]
      2) CLI override (applies to every selected exp, ignoring per-exp values):
             python plot.py --index-filter 9,16,37

Usage (defaults render every exp in `exps`):
    python plot.py
    python plot.py --only newVAE_vs_oldVAE,t2va_single_captions_300_5s_15s_first100
    python plot.py --exclude-steps 30000
    python plot.py --exclude newVAE_yesRescale:30000,oldVAE_yesRescale:40000
    python plot.py --only t2va_single_captions_300_5s_15s_first100 --index-filter 9,16
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------------------------------
# Data sources: variant name -> list of asr_results root directories.
# Each root contains `iter_<step>/wer_results.json` files.
# --------------------------------------------------------------------------
info = {
    "fl2va_single_captions_300_5s_15s_first100": ["/apdcephfs_wzd2/share_305640887/hunyuan/jarvizhang/basic_train_jarvizhang_20260522022116_46e15478/95a68c749e885016019e89287b150084/samples_metrics/asr_results/fl2va_single_captions_300_5s_15s_first100",],
    "fl2va_multi_captions_300_5s_15s_first100": ["/apdcephfs_wzd2/share_305640887/hunyuan/jarvizhang/basic_train_jarvizhang_20260522022116_46e15478/95a68c749e885016019e89287b150084/samples_metrics/asr_results/fl2va_multi_captions_300_5s_15s_first100",],
    "t2va_single_captions_300_5s_15s_first100": ["/apdcephfs_wzd2/share_305640887/hunyuan/jarvizhang/basic_train_jarvizhang_20260522022116_46e15478/95a68c749e885016019e89287b150084/samples_metrics/asr_results/t2va_single_captions_300_5s_15s_first100",],
    "t2va_multi_captions_300_5s_15s_first100": ["/apdcephfs_wzd2/share_305640887/hunyuan/jarvizhang/basic_train_jarvizhang_20260522022116_46e15478/95a68c749e885016019e89287b150084/samples_metrics/asr_results/t2va_multi_captions_300_5s_15s_first100",],
    "i2va_single_captions_300_5s_15s_first100": ["/apdcephfs_wzd2/share_305640887/hunyuan/jarvizhang/basic_train_jarvizhang_20260522022116_46e15478/95a68c749e885016019e89287b150084/samples_metrics/asr_results/i2va_single_captions_300_5s_15s_first100",],
    "i2va_multi_captions_300_5s_15s_first100": ["/apdcephfs_wzd2/share_305640887/hunyuan/jarvizhang/basic_train_jarvizhang_20260522022116_46e15478/95a68c749e885016019e89287b150084/samples_metrics/asr_results/i2va_multi_captions_300_5s_15s_first100",],

    "newVAE_yesRescale": ["/apdcephfs_wzd2/share_305640887/2_public_experiments/Leo2.0_ablation/only_train_t2i_t2a_ablation_v2/1_newVAE_yesRescale_new/samples_metrics/asr_results/valid_sample_300",],
    "oldVAE_yesRescale": ["/apdcephfs_wzd2/share_305640887/2_public_experiments/Leo2.0_ablation/only_train_t2i_t2a_ablation_v2/0_oldVAE_yesRescale_new/samples_metrics/asr_results/valid_sample_300",],
}

BASE_OUTPUT_DIR = "/apdcephfs_wzd2/share_305640887/2_public_experiments/Leo2.0_ablation/only_train_t2i_t2a_ablation_v2/metrics"

exps = {
    "newVAE_vs_oldVAE": {
        "info": ["newVAE_yesRescale", "oldVAE_yesRescale"],
        "output": f"{BASE_OUTPUT_DIR}/newVAE_vs_oldVAE.png"
    },

    # "t2va_single_captions_300_5s_15s_first100": {
    #     "info": ["t2va_single_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/t2va_single_captions_300_5s_15s_first100.png"
    # },
    # "t2va_multi_captions_300_5s_15s_first100": {
    #     "info": ["t2va_multi_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/t2va_multi_captions_300_5s_15s_first100.png"
    # },
    # "i2va_single_captions_300_5s_15s_first100": {
    #     "info": ["i2va_single_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/i2va_single_captions_300_5s_15s_first100.png"
    # },
    # "i2va_multi_captions_300_5s_15s_first100": {
    #     "info": ["i2va_multi_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/i2va_multi_captions_300_5s_15s_first100.png"
    # },
    # "fl2va_single_captions_300_5s_15s_first100": {
    #     "info": ["fl2va_single_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/fl2va_single_captions_300_5s_15s_first100.png"
    # },
    # "fl2va_multi_captions_300_5s_15s_first100": {
    #     "info": ["fl2va_multi_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/fl2va_multi_captions_300_5s_15s_first100.png"
    # },
    #
    #
    #
    # "t2va_single_captions_300_5s_15s_first100_filtered": {
    #     "info": ["t2va_single_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/t2va_single_captions_300_5s_15s_first100_filtered.png",
    #     "index_filter": ['9', '16', '18', '28', '38', '54', '58', '73', '95', '97', '99', '0', '4', '7', '13', '14', '21', '22', '26', '30', '32', '35', '36', '37', '39', '40', '42', '45', '52', '55', '63', '71', '75', '76', '80', '83', '84', '85'],
    # },
    # "t2va_multi_captions_300_5s_15s_first100_filtered": {
    #     "info": ["t2va_multi_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/t2va_multi_captions_300_5s_15s_first100_filtered.png",
    #     "index_filter": ['1', '2', '4', '5', '6', '8', '9', '12', '17', '19', '21', '27', '28', '30', '34', '35', '36', '39', '40', '45', '46', '53', '57', '60', '61', '62', '64', '65', '69', '72', '91', '93', '94', '95', '96', '98', '99', '0', '7', '13', '14', '15', '42', '44', '48', '52', '54', '71', '73', '78', '83'],
    # },
    # "i2va_single_captions_300_5s_15s_first100_filtered": {
    #     "info": ["i2va_single_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/i2va_single_captions_300_5s_15s_first100_filtered.png",
    #     "index_filter": ['9', '16', '18', '28', '38', '54', '58', '73', '95', '97', '99', '0', '4', '7', '13', '14', '21', '22', '26', '30', '32', '35', '36', '37', '39', '40', '42', '45', '52', '55', '63', '71', '75', '76', '80', '83', '84', '85'],
    # },
    # "i2va_multi_captions_300_5s_15s_first100_filtered": {
    #     "info": ["i2va_multi_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/i2va_multi_captions_300_5s_15s_first100_filtered.png",
    #     "index_filter": ['1', '2', '4', '5', '6', '8', '9', '12', '17', '19', '21', '27', '28', '30', '34', '35', '36', '39', '40', '45', '46', '53', '57', '60', '61', '62', '64', '65', '69', '72', '91', '93', '94', '95', '96', '98', '99', '0', '7', '13', '14', '15', '42', '44', '48', '52', '54', '71', '73', '78', '83'],
    # },
    # "fl2va_single_captions_300_5s_15s_first100_filtered": {
    #     "info": ["fl2va_single_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/fl2va_single_captions_300_5s_15s_first100_filtered.png",
    #     "index_filter": ['9', '16', '18', '28', '38', '54', '58', '73', '95', '97', '99', '0', '4', '7', '13', '14', '21', '22', '26', '30', '32', '35', '36', '37', '39', '40', '42', '45', '52', '55', '63', '71', '75', '76', '80', '83', '84', '85'],
    # },
    # "fl2va_multi_captions_300_5s_15s_first100_filtered": {
    #     "info": ["fl2va_multi_captions_300_5s_15s_first100"],
    #     "output": f"{BASE_OUTPUT_DIR}/fl2va_multi_captions_300_5s_15s_first100_filtered.png",
    #     "index_filter": ['1', '2', '4', '5', '6', '8', '9', '12', '17', '19', '21', '27', '28', '30', '34', '35', '36', '39', '40', '45', '46', '53', '57', '60', '61', '62', '64', '65', '69', '72', '91', '93', '94', '95', '96', '98', '99', '0', '7', '13', '14', '15', '42', '44', '48', '52', '54', '71', '73', '78', '83'],
    # },
}


# --------------------------------------------------------------------------
# Visual style (mirrors vis/plots_CLIP_FID_HPSv2_metrics_utils.py)
# --------------------------------------------------------------------------
_PALETTE = (
    "#3A86FF",  # bright blue
    "#FB5607",  # vibrant orange
    "#8338EC",  # vivid purple
    "#06A77D",  # teal green
    "#FFBE0B",  # amber
    "#FF006E",  # magenta
    "#118AB2",  # ocean blue
    "#73A580",  # sage
)
_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")

_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "axes.labelpad": 8,
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.facecolor": "white",
    "legend.edgecolor": "#CCCCCC",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}

ITER_RE = re.compile(r"^iter_(\d+)$")
WER_JSON_NAME = "wer_results.json"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _format_wer(value: float) -> str:
    """Format WER as a percentage with 2 decimal places (e.g. 6.51%)."""
    return f"{value * 100:.2f}%"


def _load_wer_record(path: str) -> dict:
    """Parse and lightly validate a wer_results.json; return the raw dict.

    Aggregation (overall vs index-filtered mean) is done by `_aggregate_wer`
    so the same parsed record can be reused across multiple exps with
    different filters.
    """
    # Some upstream writers emit the JSON with a UTF-8 BOM, which stdlib
    # `json` refuses to parse; `utf-8-sig` transparently strips it.
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON structure in {path}: {type(data).__name__}")
    if "overall_wer" not in data:
        raise KeyError(f"`overall_wer` missing in {path}")
    return data


def _aggregate_wer(
    record: dict,
    index_filter: Optional[Set[str]] = None,
    *,
    json_path: str = "",
) -> Tuple[Optional[float], Optional[int]]:
    """Reduce a wer_results.json record to a single (wer, n) pair.

    - `index_filter` is `None` / empty: return the upstream `overall_wer`
      and `num_samples` (existing behaviour).
    - `index_filter` is set: return (mean of per-sample `wer` over `details`
      entries whose stringified `index` is in the filter, count of matched
      entries). Returns (None, 0) when no detail matches so the caller can
      skip the step. Missing-index warnings reference `json_path` if given.

    NB: the per-detail `wer` field is already a per-sample WER, so we just
    average it. This is not a word-weighted aggregate (we don't have word
    counts in `details`), but it matches what the user asked for: "对指定
    index 子集求平均 wer".
    """
    if not index_filter:
        wer = float(record["overall_wer"])
        n = record.get("num_samples")
        if n is not None:
            try:
                n = int(n)
            except (TypeError, ValueError):
                n = None
        return wer, n

    details = record.get("details")
    if not isinstance(details, list):
        raise KeyError(
            f"`details` missing or not a list in {json_path!r}; "
            f"cannot apply index_filter"
        )

    matched_wers: List[float] = []
    matched_indices: Set[str] = set()
    for d in details:
        idx = str(d.get("index"))
        if idx in index_filter:
            try:
                w = float(d["wer"])
            except (TypeError, KeyError, ValueError):
                continue
            matched_wers.append(w)
            matched_indices.add(idx)

    if not matched_wers:
        return None, 0

    missing = index_filter - matched_indices
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        loc = f" in {json_path}" if json_path else ""
        print(
            f"[WARN]{loc}: index_filter missing "
            f"{len(missing)}/{len(index_filter)} indices: {sample}{more}"
        )

    return sum(matched_wers) / len(matched_wers), len(matched_wers)


def collect_variant_records(
    variant_name: str,
    roots: List[str],
    excluded_steps: Optional[Set[int]] = None,
) -> List[Tuple[int, dict, str]]:
    """Return sorted [(step, raw_record, json_path), ...] for one variant.

    Walks every root in `roots`, finds `iter_<step>` subdirs, reads
    `wer_results.json` from each. The user has confirmed iters never collide
    across the roots, so we don't bother deduplicating — but we do warn if a
    collision is seen anyway, to avoid silent data loss.

    Aggregation/filtering is intentionally split out into `derive_curve` so
    the same parsed records can be reused by multiple exps that apply
    different `index_filter`s.
    """
    excluded_steps = excluded_steps or set()
    seen_steps: Dict[int, str] = {}
    records: List[Tuple[int, dict, str]] = []

    if not roots:
        print(f"[WARN] no roots configured for `{variant_name}`")
        return records

    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] missing directory for `{variant_name}`: {root}")
            continue
        for entry in sorted(os.listdir(root)):
            m = ITER_RE.match(entry)
            if not m:
                continue
            step = int(m.group(1))
            if step in excluded_steps:
                print(f"[INFO] {variant_name}: excluding step {step}")
                continue
            json_path = os.path.join(root, entry, WER_JSON_NAME)
            if not os.path.isfile(json_path):
                # Evaluation for this iter may still be in flight; skip silently.
                continue
            try:
                record = _load_wer_record(json_path)
            except Exception as exc:
                print(f"[WARN] {variant_name} step {step}: {exc}")
                continue
            if step in seen_steps:
                print(
                    f"[WARN] {variant_name}: duplicate step {step} in "
                    f"{seen_steps[step]} and {root}; keeping the first occurrence"
                )
                continue
            seen_steps[step] = root
            records.append((step, record, json_path))

    records.sort(key=lambda p: p[0])
    return records


def derive_curve(
    variant_name: str,
    records: List[Tuple[int, dict, str]],
    index_filter: Optional[Set[str]] = None,
) -> List[Tuple[int, float, Optional[int]]]:
    """Apply optional index filter to raw records → plottable curve points."""
    points: List[Tuple[int, float, Optional[int]]] = []
    for step, record, json_path in records:
        try:
            wer, n = _aggregate_wer(record, index_filter, json_path=json_path)
        except Exception as exc:
            print(f"[WARN] {variant_name} step {step}: {exc}")
            continue
        if wer is None:
            # No detail matched the filter at this step; skip silently on the
            # curve (the per-step warn above already noted missing indices).
            print(
                f"[INFO] {variant_name} step {step}: no detail matched "
                f"index_filter, skipping"
            )
            continue
        points.append((step, wer, n))
    return points


# --------------------------------------------------------------------------
# CLI argument parsing helpers
# --------------------------------------------------------------------------
def parse_exclusions(
    exclude_steps_arg: Optional[str],
    exclude_arg: Optional[str],
    variant_names: List[str],
) -> Dict[str, Set[int]]:
    """Build a {variant_name -> {steps to skip}} mapping from CLI args.

    --exclude-steps "30000,40000"          excludes those steps from ALL variants.
    --exclude "newVAE_yesRescale:30000,oldVAE_yesRescale:40000"   per-variant.
    The two flags compose; per-variant entries add on top of global ones.
    """
    out: Dict[str, Set[int]] = {name: set() for name in variant_names}

    if exclude_steps_arg:
        global_steps = {int(s) for s in exclude_steps_arg.split(",") if s.strip()}
        for name in variant_names:
            out[name].update(global_steps)

    if exclude_arg:
        for token in exclude_arg.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(
                    f"--exclude entry `{token}` must look like `variant:step`"
                )
            variant, step_str = token.split(":", 1)
            variant = variant.strip()
            step_str = step_str.strip()
            if variant not in out:
                # The variant may simply not be referenced by the exp being
                # plotted; record it anyway so it kicks in if/when included.
                out[variant] = set()
            out[variant].add(int(step_str))

    return out


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _format_index_filter(index_filter: Optional[Set[str]]) -> str:
    """Render an index filter compactly for plot titles."""
    if not index_filter:
        return ""
    # Sort numerically when all entries look like ints, else lexicographically.
    try:
        items = sorted(index_filter, key=lambda s: int(s))
    except ValueError:
        items = sorted(index_filter)
    if len(items) <= 6:
        return ",".join(items)
    head = ",".join(items[:5])
    return f"{head},...(+{len(items) - 5})"


def plot_exp(
    exp_name: str,
    curves: Dict[str, List[Tuple[int, float, Optional[int]]]],
    out_path: str,
    index_filter: Optional[Set[str]] = None,
) -> None:
    """Render one experiment's step-vs-WER curves with per-point value labels."""
    with plt.rc_context(_RC_PARAMS):
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        ax.set_facecolor("#FAFBFC")

        plotted_any = False
        sample_counts: Set[int] = set()
        for idx, (variant_name, points) in enumerate(curves.items()):
            if not points:
                print(f"[WARN] {exp_name}: no points for variant `{variant_name}`")
                continue
            color = _PALETTE[idx % len(_PALETTE)]
            marker = _MARKERS[idx % len(_MARKERS)]
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            for _, _, n in points:
                if n is not None:
                    sample_counts.add(n)

            ax.plot(
                steps, values,
                color=color,
                marker=marker,
                markersize=7,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.6,
                linewidth=2.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=variant_name,
                zorder=3,
            )

            # Alternate annotation side per variant so labels from different
            # curves at the same step don't collide visually.
            place_above = (idx % 2 == 0)
            offset = (0, 9) if place_above else (0, -11)
            va = "bottom" if place_above else "top"
            for step, value, _ in points:
                ax.annotate(
                    _format_wer(value),
                    xy=(step, value),
                    xytext=offset,
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=8.5,
                    color=color,
                    zorder=4,
                    bbox=dict(
                        boxstyle="round,pad=0.22",
                        facecolor="white",
                        edgecolor=color,
                        linewidth=0.5,
                        alpha=0.9,
                    ),
                )
            plotted_any = True

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if sample_counts:
            if len(sample_counts) == 1:
                n_str = f"n={next(iter(sample_counts))}"
            else:
                n_str = f"n={min(sample_counts)}-{max(sample_counts)}"
            title_extra = n_str
        else:
            title_extra = ""
        if index_filter:
            filt_str = f"idx=[{_format_index_filter(index_filter)}], mean(per-sample wer)"
            title_extra = f"{title_extra}, {filt_str}" if title_extra else filt_str
        if title_extra:
            title = f"WER across training steps  ({exp_name}, {title_extra})"
        else:
            title = f"WER across training steps  ({exp_name})"
        ax.set_title(title, color="#1A1A1A")
        ax.set_xlabel("training step")
        ax.set_ylabel("WER")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
        ax.grid(False, axis="x")
        ax.margins(x=0.05, y=0.22)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f}%"))

        if plotted_any:
            ax.legend(loc="best", fancybox=True)
        else:
            ax.text(
                0.5, 0.5,
                "No data points found",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=12, color="#888888",
            )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    print(f"saved {out_path}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot WER curves across training steps for ASR eval runs.",
    )
    parser.add_argument(
        "--only",
        default="",
        help=(
            "Comma-separated subset of exp names from `exps` to render. "
            "When empty (default), every exp in `exps` is rendered."
        ),
    )
    parser.add_argument(
        "--exclude-steps",
        default="",
        help=(
            "Comma-separated step numbers to skip across ALL variants, "
            "e.g. `--exclude-steps 30000,40000`."
        ),
    )
    parser.add_argument(
        "--exclude",
        default="",
        help=(
            "Comma-separated `variant:step` entries to skip per-variant, "
            "e.g. `--exclude newVAE_yesRescale:30000`."
        ),
    )
    parser.add_argument(
        "--index-filter",
        default="",
        help=(
            "Comma-separated sample index whitelist (e.g. `--index-filter 9,16`). "
            "When set, every selected exp uses the simple mean of per-sample "
            "`wer` over `details` whose `index` falls in this list, instead of "
            "the upstream `overall_wer`. Overrides per-exp `index_filter` in "
            "the `exps` dict."
        ),
    )
    args = parser.parse_args()

    if args.only:
        wanted = [name.strip() for name in args.only.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in exps]
        if unknown:
            raise SystemExit(f"--only refers to unknown exp(s): {unknown}")
        selected_exps = {name: exps[name] for name in wanted}
    else:
        selected_exps = dict(exps)

    referenced_variants = sorted({
        v for cfg in selected_exps.values() for v in cfg["info"]
    })
    missing = [v for v in referenced_variants if v not in info]
    if missing:
        raise SystemExit(
            f"exp(s) reference variants missing from `info`: {missing}"
        )

    exclusions = parse_exclusions(
        exclude_steps_arg=args.exclude_steps,
        exclude_arg=args.exclude,
        variant_names=referenced_variants,
    )

    cli_filter: Optional[Set[str]] = None
    if args.index_filter.strip():
        cli_filter = {tok.strip() for tok in args.index_filter.split(",") if tok.strip()}
        if not cli_filter:
            raise SystemExit("--index-filter parsed to an empty set")

    # Load raw records once per variant (parsing details JSON is the costly
    # step). The same records may then be aggregated by `derive_curve` with
    # different filters for different exps.
    variant_records: Dict[str, List[Tuple[int, dict, str]]] = {
        name: collect_variant_records(name, info[name], excluded_steps=exclusions.get(name))
        for name in referenced_variants
    }

    for exp_name, cfg in selected_exps.items():
        # CLI filter wins over per-exp config.
        exp_filter_raw = cli_filter if cli_filter is not None else cfg.get("index_filter")
        exp_filter: Optional[Set[str]] = (
            {str(x) for x in exp_filter_raw} if exp_filter_raw else None
        )
        if exp_filter is not None:
            print(
                f"[INFO] {exp_name}: applying index_filter "
                f"({len(exp_filter)} indices) → mean of per-sample wer"
            )
        curves = {
            v: derive_curve(v, variant_records[v], index_filter=exp_filter)
            for v in cfg["info"]
        }
        plot_exp(exp_name, curves, cfg["output"], index_filter=exp_filter)


if __name__ == "__main__":
    main()
