import os
from typing import Union, List, Optional

import insightface
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image


class FaceSimilarityRewardModel(object):
    def __init__(self, model_name='buffalo_l', device='cpu', det_size=(640, 640), http_proxy=None, https_proxy=None):
        """Initialize Face Similarity Reward Model"""
        self.device = device
        self.model_name = model_name
        self.det_size = det_size
        self.model = self.build_reward_model()
        if http_proxy:
            os.environ['http_proxy'] = http_proxy
        if https_proxy:
            os.environ['https_proxy'] = https_proxy

    def build_reward_model(self):
        """Build and prepare the face analysis model"""
        # Set providers based on device
        if self.device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        model = insightface.app.FaceAnalysis(name=self.model_name, providers=providers)
        model.prepare(ctx_id=0, det_size=self.det_size)
        return model

    def _extract_face_embedding(self, image: Image.Image):
        """Extract face embedding from image"""
        # rgba/rgb to bgr
        image = np.array(image)[..., :3]
        image = image[..., ::-1]     # to BGR
        
        # Detect faces
        faces = self.model.get(image)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        
        # Return the first face embedding
        return faces[0].normed_embedding

    def __call__(
            self,
            images: Union[Image.Image, List[Image.Image]],
            prompts: Optional[Union[str, List[str]]] = None,
            reference_images: Union[Image.Image, List[Image.Image]] = None,
    ) -> tuple[List[float], List[int]]:
        """
        Calculate face similarity scores
        
        Args:
            images: Generated images to evaluate
            prompts: Text prompts (not used in face similarity, kept for interface compatibility)
            reference_images: Reference images to compare against. If None, will use the first image as reference
            
        Returns:
            List of similarity scores
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(reference_images, Image.Image):
            reference_images = [reference_images]
        
        assert len(images) == len(reference_images), "images and reference_images must have the same length"
        
        rewards = []
        successes = []
        for image, ref_image in zip(images, reference_images):
            try:
                # Extract face embeddings
                embedding1 = self._extract_face_embedding(image)
                embedding2 = self._extract_face_embedding(ref_image)
                
                # Calculate cosine similarity
                similarity = 1 - cosine(embedding1, embedding2)
                rewards.append(float(similarity))
                successes.append(1)
            except ValueError as e:
                # If face detection fails, return a low similarity score
                print(f"Face detection failed: {e}")
                rewards.append(0.0)
                successes.append(0)

        return rewards, successes
    

if __name__ == "__main__":
    model = FaceSimilarityRewardModel(http_proxy='http://star-proxy.oa.com:3128', https_proxy='http://star-proxy.oa.com:3128')
    ref_img_paths = [
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face1_1.png",
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face2_1.png",
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face3_1.png",
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face4_1.png",
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face5_1.png",
    ]
    img_paths = [
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face1_2.png",
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face2_2.png",
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face3_2.png",
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face4_2.png",
        "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/face_imgs/face5_2.png",
    ]
    ref_images = [Image.open(ref_img_path) for ref_img_path in ref_img_paths]
    img_images = [Image.open(img_path) for img_path in img_paths]
    rewards, successes = model(images=img_images, reference_images=ref_images)
    print(rewards, successes)
    

# pip install insightface
# PYTHONPATH=./ python3 hymm/models/reward_models/face_similarity.py
