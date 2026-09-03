#!/usr/bin/env python3
"""Evaluate one Leo2 cache case with VideoScore2."""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from string import Template

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

QUERY = Template(
    """
You are an expert for evaluating AI-generated videos from three dimensions:
(1) visual quality – clarity, smoothness, artifacts;
(2) text-to-video alignment – fidelity to the prompt;
(3) physical/common-sense consistency – naturalness and physics plausibility.

Video prompt: $prompt

Please output in this format:
visual quality: <v_score>;
text-to-video alignment: <t_score>;
physical/common-sense consistency: <p_score>
"""
)
LABELS = {
    "visual_quality": ("visual quality",),
    "text_alignment": ("text-to-video alignment",),
    "physical_consistency": (
        "physical/common-sense consistency",
        "physical consistency",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--prompts-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="TIGER-Lab/VideoScore2")
    parser.add_argument("--infer-fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_scores(text: str) -> dict[str, int]:
    """Parse the three ordinal scores despite optional numbered descriptions."""
    scores = {}
    for key, aliases in LABELS.items():
        match = None
        for label in aliases:
            match = re.search(
                re.escape(label) + r"[^:\n;]*[:：]\s*\**([1-5])\b",
                text,
                flags=re.IGNORECASE,
            )
            if match is not None:
                break
        if match is None:
            raise ValueError(f"missing {aliases!r} score in model output: {text!r}")
        scores[key] = int(match.group(1))
    return scores


def find_video(root: Path, case: str, index: int) -> Path:
    """Resolve the unique generated video for one prompt index."""
    matches = sorted((root / case / "samples").glob(f"*/videos/{index}_0.mp4"))
    if len(matches) != 1:
        raise ValueError(f"expected one video for {case=} {index=}, found {matches}")
    return matches[0]


def load_prompts(path: Path) -> list[dict[str, str]]:
    """Load and validate the ordered benchmark prompts."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for expected, row in enumerate(rows):
        if int(row["index"]) != expected:
            raise ValueError(f"prompt indices must be contiguous from zero: {row}")
    return rows


def evaluate_video(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    video: Path,
    prompt: str,
    infer_fps: float,
    max_new_tokens: int,
    seed: int,
) -> tuple[dict[str, int], str]:
    """Run the official VideoScore2 prompt and return robustly parsed hard scores."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video), "fps": infer_fps},
                {"type": "text", "text": QUERY.substitute(prompt=prompt)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=infer_fps,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    input_length = inputs["input_ids"].shape[1]
    output = processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_scores(output), output


def main() -> None:
    """Load VideoScore2 once and evaluate every prompt in one cache case."""
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prompts = load_prompts(args.prompts_csv)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    completed = set()
    if args.output.exists():
        completed = {json.loads(line)["prompt_index"] for line in args.output.read_text().splitlines() if line}
        prompts = [row for row in prompts if int(row["index"]) not in completed]
    if not prompts:
        print(f"already complete: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True).to("cuda")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    with args.output.open("a" if args.resume else "x") as handle:
        for row in prompts:
            started = time.perf_counter()
            index = int(row["index"])
            video = find_video(args.benchmark_root, args.case, index)
            scores, raw_output = evaluate_video(
                model,
                processor,
                video,
                row["prompt"],
                args.infer_fps,
                args.max_new_tokens,
                int(row["seed"]),
            )
            result = {
                "case": args.case,
                "prompt_index": index,
                "generation_seed": int(row["seed"]),
                "evaluator_seed": int(row["seed"]),
                "prompt": row["prompt"],
                "video": str(video),
                "model": args.model,
                "infer_fps": args.infer_fps,
                "elapsed_seconds": time.perf_counter() - started,
                **scores,
                "raw_output": raw_output,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({k: result[k] for k in ("case", "prompt_index", *LABELS)}))


if __name__ == "__main__":
    main()
