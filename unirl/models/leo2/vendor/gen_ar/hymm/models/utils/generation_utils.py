import json
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers.generation.utils import GenerateOutput
from PIL import Image

from hymm.utils.file_utils import save_to_csv
from processors.video_kits import save_audio, save_video_audio


@dataclass
class MultimodalGenerationOutputs:
    batch: Optional[dict[str, Any]] = None
    texts: Optional[GenerateOutput | torch.LongTensor | list[str]] = None
    images: Optional[torch.Tensor | list[Image.Image]] = None
    videos: Optional[torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray]] = None
    audios: Optional[torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray]] = None

    def is_empty(self):
        return (
            self.texts is None and self.images is None and self.videos is None and self.audios is None
        ) or (self.batch is not None and len(self.batch["index"]) == 0)

    def postprocess_outputs(self, batch: dict[str, Any]):
        """ Check and remove dummy samples from batch, results and images."""
        batch_size = len(batch["index"])
        valid_indices = torch.where(batch["is_dummy"].logical_not())[0].tolist()
        if len(valid_indices) == 0:
            return MultimodalGenerationOutputs(batch={"index": []})

        valid_batch = {
            k: v[valid_indices]
            if isinstance(v, torch.Tensor)
            else [v[i] for i in valid_indices]
            for k, v in batch.items()
        }

        def assert_length(name, data):
            assert len(data) == batch_size, f"Length of {name}({len(data)}) must match batch size({batch_size})."

        if self.texts is not None:
            if isinstance(self.texts, list):
                assert_length("texts", self.texts)
                valid_texts = [self.texts[vi] for vi in valid_indices]
            else:
                self.texts.sequences = [self.texts.sequences[vi] for vi in valid_indices]    # type: ignore
                self.texts.logits = tuple([logit[valid_indices] for logit in self.texts.logits])  # type: ignore
                valid_texts = self.texts
        else:
            valid_texts = None

        if self.images is not None:
            assert_length("images", self.images)
            if isinstance(self.images, torch.Tensor):
                valid_images = self.images[valid_indices]
            else:
                valid_images = [self.images[vi] for vi in valid_indices]
        else:
            valid_images = None

        if self.videos is not None:
            assert_length("videos", self.videos)
            if isinstance(self.videos, torch.Tensor):
                valid_videos = self.videos[valid_indices]
            else:
                valid_videos = [self.videos[vi] for vi in valid_indices]
        else:
            valid_videos = None

        if self.audios is not None:
            assert_length("audios", self.audios)
            if isinstance(self.audios, torch.Tensor):
                valid_audios = self.audios[valid_indices]
            else:
                valid_audios = [self.audios[vi] for vi in valid_indices]
        else:
            valid_audios = None

        return MultimodalGenerationOutputs(
            batch=valid_batch,
            texts=valid_texts,
            images=valid_images,
            videos=valid_videos,
            audios=valid_audios,
        )

    def postprocess_images(self, save_images: bool, image_processor=None):
        if not save_images:
            self.images = None

        if self.images is not None:
            assert image_processor is not None, "image_processor must be provided for image postprocessing."
            self.images = image_processor.pt_to_numpy(self.images)
            self.images = image_processor.numpy_to_pil(self.images)

    @staticmethod
    def serialize_data(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().tolist()
        elif isinstance(data, list):
            return [
                json.dumps(d, ensure_ascii=False) if isinstance(d, (list, dict)) else d
                for d in data
            ]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def save_to(self, save_base: Path, summary_file_name: str, **kwargs):
        if self.is_empty():
            return

        # Sanity check
        if self.videos is not None:
            assert "fps" in kwargs, "`fps` must be provided for video saving."
        if self.audios is not None:
            assert "sample_rate" in kwargs, "`sample_rate` must be provided for audio saving."

        # Build saving response structure
        response = [dict(role="assistant", content=[]) for _ in self.batch["index"]]

        if self.texts is not None:
            for i, text in enumerate(self.texts):
                response[i]["content"].append(dict(type="text", text=text))

        if self.images is not None:
            save_image_base = save_base / "images"
            save_image_base.mkdir(parents=True, exist_ok=True)
            self.batch["gen_images"] = []
            for i, image in enumerate(self.images):
                image_path = save_image_base / f"{self.batch['index'][i]}_0.png"
                image.save(image_path)
                self.batch["gen_images"].append(str(image_path))
                response[i]["content"].append(dict(type="image", image_path=str(image_path)))

        if self.videos is not None:
            save_video_base = save_base / "videos"
            save_video_base.mkdir(parents=True, exist_ok=True)
            self.batch["gen_videos"] = []

            _audios = [None] * len(self.videos) if self.audios is None else self.audios
            for i, (video, _audio) in enumerate(zip(self.videos, _audios)):
                video_path = save_video_base / f"{self.batch['index'][i]}_0.mp4"
                save_kwargs = dict(fps=kwargs["fps"])
                if _audio is not None:
                    assert kwargs["sample_rate"] is not None, "`sample_rate` must be provided for video audio saving."
                    save_kwargs["sample_rate"] = kwargs["sample_rate"]
                save_video_audio(video, _audio, video_path, **save_kwargs)
                self.batch["gen_videos"].append(str(video_path))
                response[i]["content"].append(dict(type="video", video_path=str(video_path)))

        # Only audio generation without videos.
        if self.videos is None and self.audios is not None:
            assert kwargs["sample_rate"] is not None, "`sample_rate` must be provided for audio saving."
            save_audio_base = save_base / "audios"
            save_audio_base.mkdir(parents=True, exist_ok=True)
            self.batch["gen_audios"] = []
            for i, audio in enumerate(self.audios):
                audio_path = save_audio_base / f"{self.batch['index'][i]}_0.wav"
                save_audio(audio, audio_path, sample_rate=kwargs["sample_rate"])
                self.batch["gen_audios"].append(str(audio_path))
                response[i]["content"].append(dict(type="audio", audio_path=str(audio_path)))

        self.batch["response"] = response

        # Save batch results to csv
        save_results_path = save_base / summary_file_name
        df = pd.DataFrame({k: self.serialize_data(v) for k, v in self.batch.items()})
        save_to_csv(df, save_results_path, append=True)
