"""Benchmark-only consolidated CLIP + PickScore HTTP service.

Loads one CLIP-compatible model and derives both reward formulas from the same
image/text embeddings. This is the consolidated-process control for the MPS
experiment in issue #463; it is not a production reward-service backend.
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor


class _Turn(BaseModel):
    text: str
    image_b64: str | None = None


class _Request(BaseModel):
    history: list[_Turn]
    required_rewards: list[str]
    metadata: dict[str, Any] | None = None


class _Batch(BaseModel):
    requests: list[_Request]


class _ConsolidatedScorer:
    def __init__(self, weights: str, dtype: str) -> None:
        torch_dtype = getattr(torch, dtype)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            CLIPModel.from_pretrained(weights)
            .to(self.device, dtype=torch_dtype)
            .eval()
        )
        self.model.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(weights)
        # Transformers tokenizers/model forwards are not guaranteed thread-safe.
        self.lock = threading.Lock()

    @staticmethod
    def _tensor(value: Any) -> torch.Tensor:
        return getattr(value, "pooler_output", value)

    @torch.inference_mode()
    def score(
        self, prompts: list[str], images: list[Image.Image]
    ) -> list[tuple[float, float]]:
        with self.lock:
            image_inputs = self.processor(
                images=images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = self.processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = {
                key: value.to(self.device) for key, value in image_inputs.items()
            }
            text_inputs = {
                key: value.to(self.device) for key, value in text_inputs.items()
            }
            image_embs = self._tensor(
                self.model.get_image_features(**image_inputs)
            )
            text_embs = self._tensor(
                self.model.get_text_features(**text_inputs)
            )
            image_embs = image_embs / image_embs.norm(
                p=2, dim=-1, keepdim=True
            )
            text_embs = text_embs / text_embs.norm(
                p=2, dim=-1, keepdim=True
            )
            similarity = (
                self.model.logit_scale.exp()
                * (text_embs @ image_embs.T).diagonal()
            ).float().cpu()
        return [
            (float(value / 30.0), float(value / 26.0))
            for value in similarity
        ]


def _decode(turn: _Turn) -> Image.Image:
    if turn.image_b64 is None:
        raise ValueError("consolidated benchmark accepts image requests only")
    return Image.open(io.BytesIO(base64.b64decode(turn.image_b64))).convert("RGB")


def create_app(weights: str, dtype: str) -> FastAPI:
    state: dict[str, _ConsolidatedScorer] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["scorer"] = _ConsolidatedScorer(weights, dtype)
        yield
        state.clear()

    app = FastAPI(lifespan=lifespan)

    @app.get("/rewards")
    def rewards() -> dict[str, list[str]]:
        return {"rewards": ["clip", "pickscore"]}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "rewards": {
                "clip": ["consolidated:ready"],
                "pickscore": ["consolidated:ready"],
            },
        }

    @app.post("/score")
    def score(batch: _Batch) -> dict[str, Any]:
        prompts = [request.history[-1].text for request in batch.requests]
        images = [_decode(request.history[-1]) for request in batch.requests]
        values = state["scorer"].score(prompts, images)
        results = []
        for request, (clip_score, pickscore) in zip(batch.requests, values):
            result: dict[str, dict[str, float]] = {}
            if "clip" in request.required_rewards:
                result["clip"] = {"clip": clip_score}
            if "pickscore" in request.required_rewards:
                result["pickscore"] = {"pickscore": pickscore}
            results.append(result)
        return {"results": results, "errors": [{} for _ in results]}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(create_app(args.weights, args.dtype), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
