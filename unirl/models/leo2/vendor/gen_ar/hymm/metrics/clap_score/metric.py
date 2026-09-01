import torch
import torchaudio
import numpy as np
from ..base_metric import BaseMetric
from transformers import ClapModel, ClapProcessor


class ClapScoreMetric(BaseMetric):
    def __init__(self, clap_model_path, dataset_name="audioset"):
        super().__init__()
        self.dataset_name = dataset_name
        self.clap_model_path = clap_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = []
        self.target_sample_rate=48000
        self.target_duration=10
   
    def load_model(self, logger):
        self.model = ClapModel.from_pretrained(self.clap_model_path).to(self.device)
        self.processor = ClapProcessor.from_pretrained(self.clap_model_path)
        self.model.eval()
        logger.info(f"Loaded CLAP model from {self.clap_model_path}")

    @torch.no_grad()
    def process(self, audio_paths, video_paths, prompts, **kwargs):
        audio_sample, sample_rate = torchaudio.load(audio_paths[0])
        if sample_rate != self.target_sample_rate:
            audio_sample = audio_sample
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            audio_sample = resampler(audio_sample)
        audio_inputs = self.processor(
            audios=[audio_sample.squeeze().numpy()],
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True  # Pad audio to the required length, if necessary
        )
        audio_inputs = {key: value.to(self.device) for key, value in audio_inputs.items()}
        audio_feats = self.model.get_audio_features(**audio_inputs)

        text_inputs = self.processor(
            text=prompts,
            return_tensors="pt"
        )
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        text_feats = self.model.get_text_features(**text_inputs)
        clap_score = torch.nn.functional.cosine_similarity(audio_feats, text_feats, dim=1, eps=1e-8)
        self.results.append(clap_score.cpu().numpy())
        return clap_score

    def compute_metrics(self, results):
        mean_predictions = np.mean(results)
        count = len(results)
        return float(mean_predictions), int(count)
