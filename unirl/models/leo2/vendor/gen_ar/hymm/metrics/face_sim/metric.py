import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from ..base_metric import BaseMetric
from hymm.metrics.face_sim.facenet_pytorch import MTCNN, InceptionResnetV1
from hymm.constants import FACE_MODEL_PATH




class FaceSimMetric(BaseMetric):
    def __init__(
        self, 
        face_model_path=FACE_MODEL_PATH,
        dataset_name="face_sim",
        max_size=1024
    ):
        super().__init__()
        self.face_model_path = face_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["TORCH_HOME"] = self.face_model_path
        self.dataset_name = dataset_name
        self.max_size = max_size
        self.results = []
        
    def load_model(self, logger=None):
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device,
            state_dict_path=self.face_model_path
        )
        self.resnet = InceptionResnetV1(
            pretrained='vggface2',
            state_dict_path=self.face_model_path
        ).eval().to(self.device)

    def release_model(self):
        self.mtcnn = None
        self.resnet = None

    @torch.no_grad()
    def process(self, src_image, images, **kwargs):
        """
        Args:
            src_image (torch.Tensor): Source images tensor (B, C, H, W) or list of file paths
            images (torch.Tensor): Target images tensor (B, C, H, W) or list of file paths   
        """

        assert isinstance(src_image, torch.Tensor), "src_image must be a torch.Tensor"
        assert isinstance(images, torch.Tensor), "images must be a torch.Tensor"

        source_embeddings = self.image_tensor_to_embedding(src_image)
        target_embeddings = self.image_tensor_to_embedding(images)
        similarity = F.cosine_similarity(source_embeddings, target_embeddings, dim=1)

        print(f"Similarity of src_image and images in process of FaceSimMetric: {similarity}")
        self.results.append(similarity.cpu().numpy())
        return self.results

    @torch.no_grad()
    def image_tensor_to_embedding(self, image):

        B, C, H, W = image.shape
        image = image.permute(0, 2, 3, 1).cpu().numpy() # (B, C, H, W) -> (B, H, W, C)
        image = image * 255.0
        # vis RGB channel order image using cv2

        embeddings_list = []
        for i in range(B):
            x_aligned, prob = self.mtcnn(image[i:i+1, ...], return_prob=True)

            if x_aligned is not None and len(x_aligned) > 0 and x_aligned[0] is not None:
                x_aligned = x_aligned[0][None, ...].to(self.device)
                embeddings = self.resnet(x_aligned).detach().to(self.device)
                embeddings_list.append(embeddings)
            else:
                embeddings = torch.zeros(1, 512).detach().to(self.device)
                embeddings_list.append(embeddings)
        embeddings = torch.concat(embeddings_list, dim=0)
        return embeddings


    def compute_metrics(self, results):
        """Calculate average similarity score."""
        predictions = np.concatenate(results, axis=0)
        count = predictions.shape[0]
        return float(predictions.mean()), int(count)