import argparse
import csv
import json
import math
import os
import re
import sys
from typing import List, Optional, Tuple

import jiwer
import torch
from tqdm import tqdm

from qwen_asr import Qwen3ASRModel


# ============================================================
# Constants
# ============================================================
ENCODING = "utf-8-sig"

# When --ext is omitted, files are auto-probed in this order.
# Audio formats are tried first since Qwen3-ASR is an audio model and decoding
# audio is cheaper than demuxing video.
DEFAULT_AUTO_EXTS: List[str] = [
    # audio
    "wav", "flac", "mp3", "m4a", "aac", "ogg", "opus",
    # video (audio track will be demuxed/decoded by librosa/soundfile/sox)
    "mp4", "mov", "mkv", "avi", "webm", "m4v",
]


def _str2bool(v) -> bool:
    """Parse a CLI string into a bool. Accepts true/false/yes/no/1/0 (case-insensitive)."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(
        f"Boolean value expected (true/false/yes/no/1/0), got: {v!r}"
    )


def _parse_ext_arg(ext_arg: Optional[str]) -> List[str]:
    """Resolve the --ext CLI value into an ordered list of extensions to probe.

    - None / empty / 'auto' / 'any'  ->  DEFAULT_AUTO_EXTS (audio first, then video)
    - 'wav'                          ->  ['wav']
    - 'wav,mp4'                      ->  ['wav', 'mp4']  (probed in given order)
    Leading dots and surrounding whitespace are tolerated.
    """
    if ext_arg is None:
        return list(DEFAULT_AUTO_EXTS)
    s = ext_arg.strip().lower()
    if not s or s in ("auto", "any"):
        return list(DEFAULT_AUTO_EXTS)
    out: List[str] = []
    for tok in s.split(","):
        tok = tok.strip().lstrip(".")
        if tok and tok not in out:
            out.append(tok)
    return out or list(DEFAULT_AUTO_EXTS)


# ============================================================
# Text extraction helpers
# ============================================================
_TAG_PATTERN = re.compile(r"<(?:speech|lyrics)>(.*?)</(?:speech|lyrics)>", re.DOTALL)


def extract_ref_text(caption: str) -> str:
    """Extract all <speech> and <lyrics> content from a caption string.
    Returns concatenated text (space-joined), or empty string if none found.
    """
    if not caption:
        return ""
    matches = _TAG_PATTERN.findall(caption)
    return " ".join(m.strip() for m in matches if m.strip())


# ============================================================
# WER normalization
# ============================================================
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_STRIP_PUNCT = re.compile(r"[，。！？、；：“”‘’「」【】《》…—,.!?;:\"'()\[\]{}-]")


def _is_chinese(text: str) -> bool:
    """Return True if the text contains a significant proportion of CJK characters."""
    if not text:
        return False
    cjk_count = len(_CJK_RE.findall(text))
    return cjk_count > len(text) * 0.2


def normalize(text: str, for_display: bool = False) -> str:
    """Normalize text for WER computation.
    - Strip punctuation
    - Lowercase
    - For Chinese: insert spaces between every character (char-level WER)
    - For English: collapse whitespace (word-level WER)
    """
    text = _STRIP_PUNCT.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    if not for_display and _is_chinese(text):
        text = " ".join(list(text.replace(" ", "")))
    return text


# ============================================================
# Distributed helpers
# ============================================================
def get_dist_info() -> Tuple[int, int, int]:
    """Read RANK / WORLD_SIZE / LOCAL_RANK from env. Defaults to (0, 1, 0)."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def maybe_init_distributed(backend: str) -> None:
    """Initialize torch.distributed (for cross-rank barrier). Safe to call once."""
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)


def maybe_barrier() -> None:
    """Cross-rank barrier; no-op when not initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


# ============================================================
# CSV loading
# ============================================================
def load_samples_from_csv(
    csv_path: str,
    eval_dir: str,
    index_column: str,
    caption_column: str,
    suffix: str,
    ext_list: List[str],
    extract_tags: bool,
    first_n: Optional[int] = None,
    last_n: Optional[int] = None,
) -> Tuple[List[dict], int, int, List[str], dict]:
    """Read a ground-truth CSV and pair each row with a file in ``eval_dir``.

    Filename rule (auto-probed): for each candidate extension ``e`` in
    ``ext_list`` (in order), look for ``f"{row[index_column]}{suffix}.{e}"`` in
    ``eval_dir``. The first hit wins. Pass a single-element ``ext_list`` to
    force one extension.

    Row slicing: if ``first_n`` is set, only the first N CSV rows are
    considered. If ``last_n`` is set, only the last N rows. The two are
    mutually exclusive. Slicing happens BEFORE any per-row validation so
    skipped_no_text / skipped_no_file counts are scoped to the slice.

    Returns:
        samples, skipped_no_text, skipped_no_file, missing_files_examples, ext_counts
        where ``ext_counts`` maps the chosen extension to the number of matches.
    """
    samples: List[dict] = []
    skipped_no_text = 0
    skipped_no_file = 0
    missing_examples: List[str] = []
    ext_counts: dict = {}

    if first_n is not None and last_n is not None:
        raise ValueError(
            "Specify at most one of --first-n / --last-n, not both."
        )
    if first_n is not None and first_n < 0:
        raise ValueError(f"--first-n must be >= 0, got {first_n}")
    if last_n is not None and last_n < 0:
        raise ValueError(f"--last-n must be >= 0, got {last_n}")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")
    if not os.path.isdir(eval_dir):
        raise NotADirectoryError(f"Eval dir not found: {eval_dir}")

    # Pre-list the eval dir once. O(1) hash lookup per row × ext keeps it fast
    # even on slow networked filesystems.
    try:
        available = set(os.listdir(eval_dir))
    except OSError as e:
        raise OSError(f"Cannot list eval_dir {eval_dir}: {e}") from e

    with open(csv_path, "r", encoding=ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if index_column not in fieldnames:
            raise ValueError(
                f"Index column '{index_column}' not found in CSV. Columns: {fieldnames}"
            )
        if caption_column not in fieldnames:
            raise ValueError(
                f"Caption column '{caption_column}' not found in CSV. Columns: {fieldnames}"
            )
        rows = list(reader)

        if first_n is not None:
            rows = rows[:first_n]
        elif last_n is not None:
            rows = rows[-last_n:] if last_n > 0 else []

        for row in rows:
            raw_idx = row.get(index_column, "")
            idx = "" if raw_idx is None else str(raw_idx).strip()
            if not idx:
                skipped_no_text += 1
                continue

            caption = row.get(caption_column, "") or ""
            ref_text = extract_ref_text(caption) if extract_tags else caption.strip()
            if not ref_text:
                skipped_no_text += 1
                continue

            stem = f"{idx}{suffix}"
            chosen_path: Optional[str] = None
            chosen_ext: Optional[str] = None
            for e in ext_list:
                name = f"{stem}.{e}"
                if name in available:
                    chosen_path = os.path.join(eval_dir, name)
                    chosen_ext = e
                    break

            if chosen_path is None:
                skipped_no_file += 1
                if len(missing_examples) < 5:
                    # Show what we looked for to make debugging easy.
                    tried = ", ".join(f"{stem}.{e}" for e in ext_list[:6])
                    if len(ext_list) > 6:
                        tried += ", ..."
                    missing_examples.append(f"{eval_dir}/[{tried}]")
                continue

            ext_counts[chosen_ext] = ext_counts.get(chosen_ext, 0) + 1
            samples.append(
                {
                    "index": idx,
                    "file_path": chosen_path,
                    "ext": chosen_ext,
                    "ref_text": ref_text,
                }
            )

    return samples, skipped_no_text, skipped_no_file, missing_examples, ext_counts


# ============================================================
# ASR inference
# ============================================================
def run_asr(
    model: Qwen3ASRModel,
    samples: List[dict],
    batch_size: int,
    language: Optional[str],
    desc: str,
) -> List[str]:
    """Run ASR on all samples in batches, return list of hyp strings (one per sample)."""
    audio_paths = [s["file_path"] for s in samples]
    hyps: List[str] = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc=desc):
        batch = audio_paths[i : i + batch_size]
        results = model.transcribe(audio=batch, language=language)
        for r in results:
            hyps.append(r.text if r.text else "")
    return hyps


# ============================================================
# WER computation
# ============================================================
def compute_wer(refs: List[str], hyps: List[str]) -> dict:
    """Compute corpus-level and per-sample WER over normalized refs/hyps."""
    norm_refs = [normalize(r) for r in refs]
    norm_hyps = [normalize(h) for h in hyps]

    if any(r.strip() for r in norm_refs):
        # jiwer requires non-empty references at the corpus level
        nonempty = [(r, h) for r, h in zip(norm_refs, norm_hyps) if r.strip()]
        overall_wer = jiwer.wer([r for r, _ in nonempty], [h for _, h in nonempty])
    else:
        overall_wer = float("nan")

    per_sample: List[float] = []
    for ref, hyp in zip(norm_refs, norm_hyps):
        if not ref.strip():
            per_sample.append(float("nan"))
            continue
        try:
            per_sample.append(jiwer.wer(ref, hyp))
        except Exception:
            per_sample.append(float("nan"))

    return {"overall_wer": overall_wer, "per_sample_wer": per_sample}


# ============================================================
# Aggregation
# ============================================================
def _sort_key_index(item: dict):
    v = item.get("index", "")
    try:
        return (0, int(v))
    except (ValueError, TypeError):
        return (1, str(v))


def aggregate_parts(parts_dir: str, tgt_dir: str, args_snapshot: dict) -> Tuple[str, dict]:
    """Merge all rank part files in ``parts_dir`` into a single result JSON.

    Recomputes the overall WER over the full merged set (more correct than averaging
    per-shard WERs). Returns the output path and the final dict.
    """
    if not os.path.isdir(parts_dir):
        raise FileNotFoundError(f"Parts dir not found: {parts_dir}")

    part_files = sorted(
        os.path.join(parts_dir, f)
        for f in os.listdir(parts_dir)
        if f.startswith("rank_") and f.endswith(".json")
    )
    if not part_files:
        raise FileNotFoundError(f"No rank part files found in {parts_dir}")

    all_items: List[dict] = []
    for pf in part_files:
        with open(pf, "r", encoding=ENCODING) as f:
            data = json.load(f)
        all_items.extend(data.get("details", []))

    all_items.sort(key=_sort_key_index)

    refs = [it["ref"] for it in all_items]
    hyps = [it["hyp"] for it in all_items]
    wer_result = compute_wer(refs, hyps)
    for it, w in zip(all_items, wer_result["per_sample_wer"]):
        it["wer"] = w

    final = {
        "args": args_snapshot,
        "overall_wer": wer_result["overall_wer"],
        "num_samples": len(all_items),
        "num_part_files": len(part_files),
        "details": all_items,
    }

    out_path = os.path.join(tgt_dir, "wer_results.json")
    with open(out_path, "w", encoding=ENCODING) as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    return out_path, final


def _print_worst(final: dict, k: int = 5) -> None:
    items = final.get("details", [])
    indexed = sorted(
        enumerate(items),
        key=lambda kv: (
            kv[1].get("wer") if isinstance(kv[1].get("wer"), (int, float)) and not math.isnan(kv[1]["wer"]) else -1
        ),
        reverse=True,
    )
    print("  Top-5 worst samples:")
    for rank_, (_, it) in enumerate(indexed[:k]):
        w = it.get("wer")
        w_str = f"{w:.4f}" if isinstance(w, (int, float)) and not math.isnan(w) else "nan"
        print(
            f"    [{rank_ + 1}] idx={it.get('index')} | WER={w_str} | "
            f"file={os.path.basename(it.get('file_path', ''))}"
        )
        print(f"         REF: {normalize(it.get('ref', ''), for_display=True)[:120]}")
        print(f"         HYP: {normalize(it.get('hyp', ''), for_display=True)[:120]}")


# ============================================================
# Arg parsing
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    p.add_argument("--gt-csv", type=str, required=True, help="Path to ground-truth CSV.")
    p.add_argument("--eval-dir", type=str, required=True, help="Directory of audio/video files to evaluate.")
    p.add_argument("--tgt-dir", type=str, required=True, help="Output directory for results.")
    p.add_argument("--caption-column", type=str, default="prompt",
                   help="CSV column used as ground-truth caption (default: prompt).")
    p.add_argument("--index-column", type=str, default="index",
                   help="CSV column used as sample index (default: index).")
    p.add_argument("--suffix", type=str, default="",
                   help="Suffix appended to CSV index to form the filename stem, e.g. '_0' "
                        "for files like '0_0.wav'. Default: empty.")
    p.add_argument("--ext", type=str, default=None,
                   help="File extension(s) to look for, without leading dot. "
                        "If omitted (or set to 'auto'/'any'), auto-probes audio first "
                        f"then video: {','.join(DEFAULT_AUTO_EXTS)}. "
                        "Pass a single ext to force one ('wav') or a comma-separated "
                        "priority list ('wav,mp4').")
    p.add_argument("--is-extract-ref-text", type=_str2bool, default=True,
                   metavar="{true,false}",
                   help="If true (default), extract <speech>/<lyrics> tag content from the "
                        "caption column as the reference text. If false, use the whole "
                        "caption text as the reference.")
    p.add_argument("--first-n", type=int, default=None,
                   help="If set, only evaluate the first N rows of the CSV. "
                        "Mutually exclusive with --last-n.")
    p.add_argument("--last-n", type=int, default=None,
                   help="If set, only evaluate the last N rows of the CSV. "
                        "Mutually exclusive with --first-n.")

    # Model
    p.add_argument("--model-path", type=str,
                   default="/apdcephfs_wzd2/share_305640887/noaltian/asr_eval/models/Qwen3-ASR-1.7B",
                   help="Path to the Qwen3-ASR model directory.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"], help="Model dtype.")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device, e.g. cuda:0. Default: cuda:LOCAL_RANK if CUDA available, else cpu.")

    # Inference
    p.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate.")
    p.add_argument("--language", type=str, default=None,
                   help="Force language, e.g. 'Chinese'. Default: auto-detect.")

    # Distributed
    p.add_argument("--dist-backend", type=str, default="gloo", choices=["gloo", "nccl"],
                   help="torch.distributed backend used only for cross-rank barrier (default: gloo).")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip inference; only aggregate existing rank part files in <tgt-dir>/parts.")

    return p.parse_args()


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()

    os.makedirs(args.tgt_dir, exist_ok=True)
    parts_dir = os.path.join(args.tgt_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    rank, world_size, local_rank = get_dist_info()
    is_main = rank == 0

    args_snapshot = {k: v for k, v in vars(args).items()}
    args_snapshot.update({"rank": rank, "world_size": world_size, "local_rank": local_rank})

    # ---- aggregate-only short-circuit ----
    if args.aggregate_only:
        if is_main:
            out_path, final = aggregate_parts(parts_dir, args.tgt_dir, args_snapshot)
            wer = final["overall_wer"]
            wer_str = f"{wer * 100:.2f}%" if isinstance(wer, (int, float)) and not math.isnan(wer) else "nan"
            print(f"[Aggregate] {out_path}  WER={wer_str}  n={final['num_samples']}")
            _print_worst(final)
        return

    # ---- device + dtype resolution ----
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(
        f"[Setup rank={rank}/{world_size} local_rank={local_rank}] device={device} dtype={args.dtype}",
        flush=True,
    )
    ext_list = _parse_ext_arg(args.ext)
    if is_main:
        print(f"[Setup] gt_csv   = {args.gt_csv}")
        print(f"[Setup] eval_dir = {args.eval_dir}")
        print(f"[Setup] tgt_dir  = {args.tgt_dir}")
        print(f"[Setup] caption_column={args.caption_column!r}  index_column={args.index_column!r}")
        print(f"[Setup] is_extract_ref_text={args.is_extract_ref_text}")
        if args.first_n is not None:
            print(f"[Setup] row slice: first {args.first_n} rows of CSV")
        elif args.last_n is not None:
            print(f"[Setup] row slice: last {args.last_n} rows of CSV")
        else:
            print("[Setup] row slice: none (all rows)")
        if len(ext_list) == 1:
            print(f"[Setup] filename pattern: '<{args.index_column}>{args.suffix}.{ext_list[0]}'")
        else:
            print(
                f"[Setup] filename pattern: '<{args.index_column}>{args.suffix}.<ext>'  "
                f"(auto-probe order: {','.join(ext_list)})"
            )

    # ---- load CSV (on every rank; small file) ----
    if args.first_n is not None and args.last_n is not None:
        raise SystemExit("[Error] Specify at most one of --first-n / --last-n, not both.")

    samples_all, no_text, no_file, missing_examples, ext_counts = load_samples_from_csv(
        csv_path=args.gt_csv,
        eval_dir=args.eval_dir,
        index_column=args.index_column,
        caption_column=args.caption_column,
        suffix=args.suffix,
        ext_list=ext_list,
        extract_tags=args.is_extract_ref_text,
        first_n=args.first_n,
        last_n=args.last_n,
    )
    if is_main:
        ext_breakdown = (
            ", ".join(f"{e}={n}" for e, n in sorted(ext_counts.items(), key=lambda kv: -kv[1]))
            if ext_counts else "(none)"
        )
        print(
            f"[Load] {len(samples_all)} valid samples | "
            f"skipped no_text={no_text}, no_file={no_file} | "
            f"by ext: {ext_breakdown}"
        )
        if missing_examples:
            print("[Load] Examples of missing files (tried extensions in priority order):")
            for p_ in missing_examples:
                print(f"  - {p_}")

    if not samples_all:
        if is_main:
            print("[Exit] No valid samples. Nothing to do.")
        return

    # ---- round-robin shard across ranks ----
    shard = samples_all[rank::world_size]
    print(f"[Shard rank={rank}] {len(shard)} / {len(samples_all)} samples", flush=True)

    # ---- init distributed (only needed for barrier) ----
    if world_size > 1:
        try:
            maybe_init_distributed(args.dist_backend)
        except Exception as e:
            print(f"[WARN rank={rank}] Could not init torch.distributed ({e!r}); "
                  "will sync via file-system only.", flush=True)

    # ---- load model ----
    if is_main:
        print(f"[Model] Loading {args.model_path} on {device} ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    if is_main:
        print("[Model] Loaded.", flush=True)

    # ---- run ASR on this rank's shard ----
    refs = [s["ref_text"] for s in shard]
    hyps = run_asr(model, shard, args.batch_size, args.language, desc=f"rank{rank} ASR")
    wer_result = compute_wer(refs, hyps)

    details = [
        {
            "index": shard[i]["index"],
            "file_path": shard[i]["file_path"],
            "ref": refs[i],
            "hyp": hyps[i],
            "wer": wer_result["per_sample_wer"][i],
        }
        for i in range(len(shard))
    ]
    part_data = {
        "rank": rank,
        "world_size": world_size,
        "num_samples": len(shard),
        "overall_wer_shard": wer_result["overall_wer"],
        "details": details,
    }
    part_path = os.path.join(parts_dir, f"rank_{rank:04d}.json")
    with open(part_path, "w", encoding=ENCODING) as f:
        json.dump(part_data, f, ensure_ascii=False, indent=2)

    shard_wer = wer_result["overall_wer"]
    shard_wer_str = (
        f"{shard_wer * 100:.2f}%"
        if isinstance(shard_wer, (int, float)) and not math.isnan(shard_wer)
        else "nan"
    )
    print(
        f"[Done rank={rank}] wrote {part_path}  shard WER={shard_wer_str}  n={len(shard)}",
        flush=True,
    )

    # ---- sync, then aggregate on rank 0 ----
    maybe_barrier()

    if is_main:
        out_path, final = aggregate_parts(parts_dir, args.tgt_dir, args_snapshot)
        wer = final["overall_wer"]
        wer_str = (
            f"{wer * 100:.2f}%"
            if isinstance(wer, (int, float)) and not math.isnan(wer)
            else "nan"
        )
        print(f"\n[Aggregate] {out_path}")
        print(f"[Aggregate] Overall WER = {wer_str}   n = {final['num_samples']}")
        _print_worst(final)


if __name__ == "__main__":
    main()
