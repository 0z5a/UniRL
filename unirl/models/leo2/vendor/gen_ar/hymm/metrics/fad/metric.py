import torch
import torchaudio
import numpy as np

from hymm.metrics.fid.metric import calculate_frechet_distance
from ..base_metric import BaseMetric
from transformers import ClapModel, ClapProcessor
from .vggish import VGGish, vggish_params


class FADMetric(BaseMetric):
    def __init__(self, model_path, fid_target_path, vggish_pca_path=None, dataset_name="audioset", model_type="clap"):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.model_path = model_path
        self.fid_target_path = fid_target_path
        self.vggish_pca_path = vggish_pca_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = []
        if self.model_type == "clap":
            self.target_sample_rate = 48000
        else:
            self.target_sample_rate = vggish_params.SAMPLE_RATE
        self.target_duration=10
   
    def load_model(self, logger):
        if self.model_type == "clap":
            self.model = ClapModel.from_pretrained(self.model_path).to(self.device)
            self.processor = ClapProcessor.from_pretrained(self.model_path)
        else:
            self.model = VGGish(self.model_path, self.vggish_pca_path).to(self.device)
        self.model.eval()
        logger.info(f"Loaded CLAP model from {self.model_path}")

    @torch.no_grad()
    def process_clap(self, audio_paths):
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
        self.results.append(audio_feats.cpu().numpy())
    
    @torch.no_grad()
    def process_vggish(self, audio_paths):
        audio_sample, sample_rate = torchaudio.load(audio_paths[0])
        audio_sample = audio_sample.squeeze(0)
        if sample_rate != self.target_sample_rate:
            audio_sample = audio_sample
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            audio_sample = resampler(audio_sample)
        audio_sample = audio_sample.to(self.device)
        audio_feats = self.model(audio_sample)
        self.results.append(audio_feats.cpu().numpy())

    @torch.no_grad()
    def process(self, audio_paths, video_paths, prompts, **kwargs):
        if self.model_type == "clap":
            self.process_clap(audio_paths)
        else:
            self.process_vggish(audio_paths)
        return None

    def compute_metrics(self, results):
        predictions = np.concatenate(results, axis=0)
        count = predictions.shape[0]
        mu_prediction, sigma_prediction = np.mean(predictions, axis=0), np.cov(predictions, rowvar=False)
        targets = torch.load(self.fid_target_path)
        mu_target, sigma_target = targets["mu"], targets["sigma"]
        fid = calculate_frechet_distance(mu_prediction, sigma_prediction, mu_target, sigma_target)
        return float(fid), int(count)
