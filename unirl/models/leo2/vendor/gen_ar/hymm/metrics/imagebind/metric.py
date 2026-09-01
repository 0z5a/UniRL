import torch
import numpy as np
from ..base_metric import BaseMetric
from imagebind.models.imagebind_model import ImageBindModel, ModalityType
from imagebind.data import load_and_transform_audio_data, load_and_transform_video_data


class ImageBindMetric(BaseMetric):
    def __init__(self, image_bind_path, dataset_name="audioset"):
        super().__init__()
        self.dataset_name = dataset_name
        self.image_bind_path = image_bind_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cos_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.results = []
   
    def load_model(self, logger):
        model = ImageBindModel(
            vision_embed_dim=1280,
            vision_num_blocks=32,
            vision_num_heads=16,
            text_embed_dim=1024,
            text_num_blocks=24,
            text_num_heads=16,
            out_embed_dim=1024,
            audio_drop_path=0.1,
            imu_drop_path=0.7,
        )
        states = torch.load(self.image_bind_path, map_location="cpu")
        model.load_state_dict(states)
        logger.info(f"Loaded model from {self.image_bind_path}")
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def process(self, audio_paths, video_paths, prompts, **kwargs):
        inputs = {
            ModalityType.VISION: load_and_transform_video_data(video_paths, self.device),
            ModalityType.AUDIO: load_and_transform_audio_data(audio_paths, self.device),
        }
        with torch.no_grad():
            embeddings = self.model(inputs)
        vision_embs = embeddings[ModalityType.VISION]  # N x C 
        audio_embs = embeddings[ModalityType.AUDIO]  # N x C
        sims = self.cos_similarity(vision_embs, audio_embs)
        self.results.extend(sims.cpu().numpy().tolist())
        return sims

    def compute_metrics(self, results):
        mean_predictions = np.mean(results)
        count = len(results)
        return float(mean_predictions), int(count)
