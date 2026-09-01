"""
Unified reward interface for video generation models.
Provides a standard interface for all reward models including video and image rewards.
All reward functions are self-contained without external flow_grpo dependencies.
"""
import os
import subprocess
import sys
import uuid

# Add project root to PYTHONPATH if not already set
# This allows the script to be run directly from any directory
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import concurrent
import torch
import numpy as np
import io
from PIL import Image
from typing import List, Dict, Any, Tuple, Union, Callable
from collections import defaultdict

from hymm.constants import ASSETS_BASE

REWARD_BASE = os.getenv("REWARD_BASE", f"{ASSETS_BASE}/reward_models").rstrip('/')
REWARD_MODEL_PATH = {
    "videoalign": f"{REWARD_BASE}/VideoReward",
    "pickscore": f'{REWARD_BASE}/PickScore_v1',
    "pickscore_processor": f'{REWARD_BASE}/CLIP-ViT-H-14-laion2B-s32B-b79K',
    "clipL": f"{REWARD_BASE}/openai_clip-vit-large-patch14",
    "imagereward": f"{REWARD_BASE}/ImageReward-v1.0/ImageReward.pt",
    "imagereward_med_config.": f"{REWARD_BASE}/ImageReward-v1.0/med_config.json",
    "qwenvl_score": f"{REWARD_BASE}/Qwen2.5-VL-7B-Instruct",
    "hpsv3": f"{REWARD_BASE}/HPSv3/HPSv3.safetensors",
    "qwen2_vl_7b": f"{REWARD_BASE}/Qwen2-VL-7B-Instruct",
    "altclip": f"{REWARD_BASE}/AltCLIP",
    "altclip_ft_017": f"{REWARD_BASE}/reward_models/altcip_rm/rm_017_altclip_512_data_1_41_42_43_44_5_61_7_8_9_10_11_12_13_lr3e-6_step4000_bs4_gradacc2_gpu8_bt_clip0.1_pos/checkpoint-final",
    "altclip_ft_024": f"{REWARD_BASE}/reward_models/hunyuan_aes_rm/rm_024_altclip_512_data_1_41_42_43_44_5_61_7_8_9_10_11_12_13_14_to_24_lr3e-6_step8000_bs4_gradacc2_gpu8_bt_clip0.1_pos/checkpoint-final",
    "raft": os.path.expanduser(f"{REWARD_BASE}/raft/models/raft-things.pth"),
}

# ============================================================================
# Video Reward Models
# ============================================================================

def dynamic_degree_score(device, model_path=None, normalize=True, target_fps=8.0):
    """
    Dynamic Degree reward based on optical flow magnitude using RAFT model.
    This metric measures the motion intensity/speed in videos using optical flow.
    Based on VBench's dynamic_degree implementation.
    
    Args:
        device: Device to run the model on
        model_path: Path to RAFT model checkpoint. If None, uses default path.
        normalize: If True, normalize scores to [0, 1] range. Default True.
        target_fps: Target fps for frame extraction. Default 8.0 fps.
    
    Returns:
        A function that takes (video_paths, prompts, metadata) and returns (scores_dict, meta_dict)
    """
    import cv2
    import torch.nn as nn
    import torch.nn.functional as F
    from easydict import EasyDict as edict
    
    # Default model path - use path from constants.py, with fallback to cache
    if model_path is None:
        model_path = REWARD_MODEL_PATH.get("raft", "")
        # Fallback to cache path if MODEL_BASE path doesn't exist
        if not model_path or not os.path.exists(model_path):
            cache_path = os.path.expanduser("~/.cache/vbench/raft_model/models/raft-things.pth")
            if os.path.exists(cache_path):
                model_path = cache_path
            else:
                raise ValueError(
                    f"RAFT model not found at {model_path}. "
                    f"Please download the model using:\n"
                    f"  mkdir -p ~/.cache/vbench/raft_model && cd ~/.cache/vbench/raft_model && "
                    f"wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip && "
                    f"unzip models.zip && rm models.zip"
                )
    
    # Import RAFT model components from local copy
    from hymm.models.reward_models.raft import RAFT, InputPadder
    
    # Initialize RAFT model
    args = edict({
        "model": model_path,
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False
    })
    
    model = RAFT(args)
    ckpt = torch.load(model_path, map_location="cpu")
    new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(new_ckpt)
    model.to(device)
    model.eval()
    
    def get_flow_score(img, flo):
        """Calculate optical flow magnitude score from flow field."""
        # img: (1, C, H, W), flo: (1, 2, H, W)
        flo = flo[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        
        u = flo[:, :, 0]
        v = flo[:, :, 1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        # Take top 5% of flow magnitudes (robust to outliers)
        cut_index = int(h * w * 0.05)
        max_rad = np.mean(np.abs(np.sort(-rad_flat))[:cut_index])
        
        return max_rad
    
    def get_frames_from_video(video_path, interval=1):
        """Extract frames from video file."""
        frame_list = []
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return []
        
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0  # Default fps
        
        # Calculate interval based on target_fps
        interval = max(1, int(round(fps / target_fps)))
        
        frame_idx = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            
            if frame_idx % interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(device)
                frame_list.append(frame)
            
            frame_idx += 1
        
        video.release()
        return frame_list
    
    def compute_dynamic_score(video_path):
        """Compute dynamic degree score for a single video."""
        frames = get_frames_from_video(video_path)
        
        if len(frames) < 2:
            print(f"Warning: Not enough frames in video {video_path}")
            return float('nan')
        
        flow_scores = []
        
        with torch.no_grad():
            for i in range(len(frames) - 1):
                image1 = frames[i]
                image2 = frames[i + 1]
                
                # Pad images to be divisible by 8
                padder = InputPadder(image1.shape)
                image1_padded, image2_padded = padder.pad(image1, image2)
                
                # Compute optical flow
                _, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
                
                # Get flow magnitude score
                score = get_flow_score(image1_padded, flow_up)
                flow_scores.append(score)
        
        if len(flow_scores) == 0:
            return float('nan')
        
        # Return mean flow magnitude as the dynamic degree score
        mean_score = np.mean(flow_scores)
        
        # Optionally normalize to [0, 1] range
        # Based on VBench, typical threshold is around 6.0 * (scale/256)
        # We use a softer normalization that maps typical motion ranges to [0, 1]
        if normalize:
            # Empirically, flow magnitudes typically range from 0 to ~50 for fast motion
            # We use sigmoid-like normalization centered around 10
            normalized_score = 1.0 / (1.0 + np.exp(-(mean_score - 10) / 5))
            return normalized_score
        else:
            return mean_score
    
    def _fn(video_paths: List[str], prompts: List[str], metadata: List[Dict] = None):
        """
        Compute dynamic degree scores for videos.
        
        Args:
            video_paths: List of video file paths
            prompts: List of text prompts (not used, kept for API consistency)
            metadata: Optional metadata for each sample
            
        Returns:
            scores_dict: Dict with 'dynamic_degree' key containing list of scores
            meta_dict: Empty dict
        """
        if metadata is None:
            metadata = [{}] * len(video_paths)
        
        scores = []
        for video_path in video_paths:
            try:
                score = compute_dynamic_score(video_path)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Error computing dynamic degree for {video_path}: {e}")
                import traceback
                traceback.print_exc()
                scores.append(float('nan'))
        
        # Return as list (single score model format) for compatibility with training code
        # This allows dynamic_degree to work without sub_reward in config
        return scores, {}
    
    # Store model reference for offloading
    _fn._reward_model = model
    return _fn


def dynamic_degree_score_simple(device, target_fps=8.0, normalize=True):
    """
    Simplified Dynamic Degree reward using frame difference instead of RAFT.
    This is a lightweight alternative that doesn't require the RAFT model.
    Measures motion intensity by computing frame-to-frame pixel differences.
    
    Args:
        device: Device to run computations on
        target_fps: Target fps for frame extraction. Default 8.0 fps.
        normalize: If True, normalize scores to [0, 1] range. Default True.
    
    Returns:
        A function that takes (video_paths, prompts, metadata) and returns (scores_dict, meta_dict)
    """
    import cv2
    
    def get_frames_from_video(video_path):
        """Extract frames from video file."""
        frame_list = []
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return []
        
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0  # Default fps
        
        # Calculate interval based on target_fps
        interval = max(1, int(round(fps / target_fps)))
        
        frame_idx = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            
            if frame_idx % interval == 0:
                # Convert to grayscale for faster computation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_list.append(gray.astype(np.float32))
            
            frame_idx += 1
        
        video.release()
        return frame_list
    
    def compute_dynamic_score(video_path):
        """Compute dynamic degree score using frame differences."""
        frames = get_frames_from_video(video_path)
        
        if len(frames) < 2:
            print(f"Warning: Not enough frames in video {video_path}")
            return float('nan')
        
        diff_scores = []
        
        for i in range(len(frames) - 1):
            # Compute absolute difference between consecutive frames
            diff = np.abs(frames[i + 1] - frames[i])
            
            # Take mean of top 5% differences (robust to noise)
            diff_flat = diff.flatten()
            cut_index = max(1, int(len(diff_flat) * 0.05))
            top_diff = np.mean(np.sort(diff_flat)[-cut_index:])
            diff_scores.append(top_diff)
        
        if len(diff_scores) == 0:
            return float('nan')
        
        mean_score = np.mean(diff_scores)
        
        if normalize:
            # Normalize to [0, 1] using sigmoid
            # Typical frame differences range from 0 to ~100
            normalized_score = 1.0 / (1.0 + np.exp(-(mean_score - 30) / 15))
            return normalized_score
        else:
            return mean_score
    
    def _fn(video_paths: List[str], prompts: List[str], metadata: List[Dict] = None):
        """
        Compute dynamic degree scores for videos using frame differences.
        
        Args:
            video_paths: List of video file paths
            prompts: List of text prompts (not used, kept for API consistency)
            metadata: Optional metadata for each sample
            
        Returns:
            scores_dict: Dict with 'dynamic_degree' key containing list of scores
            meta_dict: Empty dict
        """
        if metadata is None:
            metadata = [{}] * len(video_paths)
        
        scores = []
        for video_path in video_paths:
            try:
                score = compute_dynamic_score(video_path)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Error computing dynamic degree for {video_path}: {e}")
                import traceback
                traceback.print_exc()
                scores.append(float('nan'))
        
        # Return as list (single score model format) for compatibility with training code
        return scores, {}
    
    return _fn


def tencent_remote_score(
    server_url: str = None,
    dimensions: List[str] = None,
    return_type: str = 'dict',
    version: str = 'v2',
):
    """
    Tencent AutoEval v2 remote reward service.
    
    Args:
        server_url: The request URL for the inference service. 
                    Defaults to 'http://qsave-v1-5-1119.polaris:80/inference/'
        dimensions: List of dimensions to evaluate, e.g., ['T2V_Overall', 'TA'].
                    Defaults to ['T2V_Overall', 'TA'] based on test examples.
        return_type: Return type, either 'dict' (default) or 'sum'
    
    Returns:
        A function that takes (video_paths, prompts, metadata) and returns (scores_dict, meta_dict)
    """
    from hymm.models.reward_models.tencent_autoeval_reward import TencentAutoEvalRewardInference

    reward_inferencer = TencentAutoEvalRewardInference(
        dimensions=dimensions,
        req_url=server_url,
        return_type=return_type,
        version=version,
    )
    
    def _fn(video_paths: List[str], prompts: List[str], metadata: List[Dict] = None):
        """
        Evaluate videos using Tencent AutoEval 1.5 service.
        
        Args:
            video_paths: List of video file paths (local paths or HTTP URLs)
            prompts: List of text prompts describing the videos
            metadata: Optional metadata for each sample (not used currently)
            
        Returns:
            scores_dict: Dict of {metric_name: List[float]} where each metric corresponds to a dimension
            meta_dict: Additional metadata (empty dict for now)
        """
        image_paths = None
        if metadata is None:
            metadata = [{}] * len(video_paths)
        else:
            if 'image_path' in metadata[0]:
                image_paths = [meta['image_path'] for meta in metadata]
        
        if len(video_paths) != len(prompts):
            raise ValueError(
                f"video_paths and prompts must have the same length. "
                f"Got {len(video_paths)} video_paths and {len(prompts)} prompts."
            )
        
        # Batch inference - process all videos at once
        all_entries = reward_inferencer.reward(video_paths, prompts, image_paths)
        
        # Convert entries to scores_dict using common utility
        scores_dict = _convert_reward_entries_to_scores_dict(all_entries)
        
        return scores_dict, {}
    
    return _fn


# ============================================================================
# Image Reward Models (Integrated from flow_grpo.rewards)
# ============================================================================

def jpeg_incompressibility():
    """JPEG incompressibility reward."""
    def _fn(images, prompts, metadata):
        images = convert_images_to_pil_list(images)
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}
    return _fn


def jpeg_compressibility():
    """JPEG compressibility reward (negative of incompressibility)."""
    jpeg_fn = jpeg_incompressibility()
    
    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta
    return _fn


def aesthetic_score(device):
    """Aesthetic score using CLIP + MLP."""
    from transformers import CLIPModel, CLIPProcessor
    import torch.nn as nn
    
    # MLP for aesthetic scoring
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )
        
        @torch.no_grad()
        def forward(self, embed):
            return self.layers(embed)
    
    clip = CLIPModel.from_pretrained(REWARD_MODEL_PATH["clipL"]).to(device)
    processor = CLIPProcessor.from_pretrained(REWARD_MODEL_PATH["clipL"])
    mlp = MLP().to(device)
    
    # Try to load pretrained aesthetic weights if available
    try:
        import os
        weight_path = os.path.join(os.path.dirname(__file__), "assets", "sac+logos+ava1-l14-linearMSE.pth")
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path)
            mlp.load_state_dict(state_dict)
    except:
        pass  # Use random weights if pretrained not available
    
    mlp.eval()
    clip.eval()
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embed = clip.get_image_features(**inputs)
            embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
            scores = mlp(embed).squeeze(1)
        
        return scores.cpu().numpy(), {}
    
    return _fn


def clip_score(device):
    """CLIP score for text-image alignment."""
    from transformers import CLIPModel, CLIPProcessor
    
    model = CLIPModel.from_pretrained(REWARD_MODEL_PATH["clipL"]).to(device)
    processor = CLIPProcessor.from_pretrained(REWARD_MODEL_PATH["clipL"])
    model.eval()
    
    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        
        # Process images
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        
        # Process text
        text_inputs = processor(text=prompts, padding='max_length', truncation=True, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, **text_inputs)
            scores = outputs.logits_per_image.diagonal() / 30
        
        return scores.cpu().numpy(), {}
    
    return _fn


def image_similarity_score(device):
    """Image similarity using CLIP embeddings."""
    from transformers import CLIPModel, CLIPProcessor
    
    model = CLIPModel.from_pretrained(REWARD_MODEL_PATH["clipL"]).to(device)
    processor = CLIPProcessor.from_pretrained(REWARD_MODEL_PATH["clipL"])
    model.eval()
    
    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        ref_pixels = processor(images=ref_images, return_tensors="pt")["pixel_values"].to(device)
        
        with torch.no_grad():
            pixel_embeds = model.get_image_features(pixel_values=pixels)
            ref_embeds = model.get_image_features(pixel_values=ref_pixels)
            
            pixel_embeds = pixel_embeds / pixel_embeds.norm(p=2, dim=-1, keepdim=True)
            ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)
            
            sim = pixel_embeds @ ref_embeds.T
            sim = torch.diagonal(sim, 0)
        
        return sim.cpu().numpy(), {}
    
    return _fn


def pickscore_score(device):
    """PickScore for image quality assessment.
    Supports both images and videos. For videos, extracts frames at 2 fps and averages frame scores.
    """
    from transformers import CLIPProcessor, CLIPModel
    
    # Default paths - users should override with their own paths
    processor_path = REWARD_MODEL_PATH["pickscore_processor"] # HuggingFace path
    model_path = REWARD_MODEL_PATH["pickscore"]
    
    processor = CLIPProcessor.from_pretrained(processor_path)
    model = CLIPModel.from_pretrained(model_path).eval().to(device)
    
    def _fn(images_or_video_paths, prompts, metadata):
        """
        Args:
            images_or_video_paths: Either:
                - torch.Tensor (NCHW) or list of PIL Images/numpy arrays for images
                - List[str] of video file paths for videos
            prompts: List of text prompts
            metadata: Optional metadata (not used)
        
        Returns:
            scores: numpy array of scores (one per prompt-image/video pair)
            meta_dict: Empty dict
        """
        # Check if input is video paths (list of strings) or images (tensor/PIL/etc)
        is_video_input = _is_video_input(images_or_video_paths)
        
        if is_video_input:
            # Process videos: extract frames and compute average scores
            video_paths = images_or_video_paths
            video_scores = []
            
            for video_path, prompt in zip(video_paths, prompts):
                try:
                    # Extract frames from video at 2 fps (2 frames per second)
                    # This provides good coverage while keeping computational cost reasonable
                    frames = _extract_frames_from_video(video_path, target_fps=2.0)
                    
                    if frames.shape[0] == 0:
                        # If no frames extracted, return NaN
                        video_scores.append(float('nan'))
                        continue
                    
                    # Convert frames tensor (T, C, H, W) to PIL Images
                    frame_pil_images = _convert_frames_to_pil_images(frames)
                    
                    # Compute scores for all frames with the same prompt
                    frame_prompts = [prompt] * len(frame_pil_images)
                    
                    # Preprocess images
                    image_inputs = processor(
                        images=frame_pil_images,
                        padding=True,
                        truncation=True,
                        max_length=77,
                        return_tensors="pt",
                    )
                    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                    
                    # Preprocess text
                    text_inputs = processor(
                        text=frame_prompts,
                        padding=True,
                        truncation=True,
                        max_length=77,
                        return_tensors="pt",
                    )
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                    
                    with torch.no_grad():
                        # Get embeddings
                        image_embs = model.get_image_features(**image_inputs)
                        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
                        
                        text_embs = model.get_text_features(**text_inputs)
                        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
                        
                        # Calculate scores
                        logit_scale = model.logit_scale.exp()
                        scores = logit_scale * (text_embs @ image_embs.T)
                        frame_scores = scores.diag() / 26  # Normalize to 0-1
                    
                    # Average frame scores to get video score
                    video_score = frame_scores.cpu().mean().item()
                    video_scores.append(video_score)
                
                except Exception as e:
                    # On error, return NaN for this video
                    import traceback
                    print(f"Warning: Error processing video {video_path}: {e}")
                    traceback.print_exc()
                    video_scores.append(float('nan'))
            
            return np.array(video_scores), {}
        
        else:
            # Process images (original logic)
            images = convert_images_to_pil_list(images_or_video_paths)
            
            # Preprocess images
            image_inputs = processor(
                images=images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            
            # Preprocess text
            text_inputs = processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                # Get embeddings
                image_embs = model.get_image_features(**image_inputs)
                image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
                
                text_embs = model.get_text_features(**text_inputs)
                text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
                
                # Calculate scores
                logit_scale = model.logit_scale.exp()
                scores = logit_scale * (text_embs @ image_embs.T)
                scores = scores.diag() / 26  # Normalize to 0-1
            
            return scores.cpu().numpy(), {}
    
    # Store model reference for offloading
    _fn._reward_model = model
    return _fn


def imagereward_score(device):
    """ImageReward score."""
    try:
        import ImageReward as RM
    except ImportError:
        raise ImportError("ImageReward package not installed. Please install with: pip install image-reward")
    
    model = RM.load(REWARD_MODEL_PATH["imagereward"], device=device, med_config=REWARD_MODEL_PATH["imagereward_med_config."]).eval()
    model.requires_grad_(False)
    
    def _fn(images, prompts, metadata):
        images = convert_images_to_pil_list(images)
        
        rewards = []
        for prompt, image in zip(prompts, images):
            _, reward = model.inference_rank(prompt, [image])
            rewards.append(reward)
        
        return np.array(rewards), {}
    
    return _fn


def qwenvl_score(device):
    """QwenVL score for aesthetic quality."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Qwen VL packages not installed")
    
    import base64
    import re
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        REWARD_MODEL_PATH["qwenvl_score"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(device)
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(REWARD_MODEL_PATH["qwenvl_score"], use_fast=True)
    
    task = '''Your role is to evaluate the aesthetic quality score of given images.
1. Bad: Extremely blurry, underexposed with significant noise, indiscernible subjects, and chaotic composition.
2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward composition with cut-off subjects.
3. Fair: In focus with adequate lighting, dull colors, decent composition but lacks creativity.
4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with a clear focal point.
5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful composition with emotional impact.

Please first provide a detailed analysis of the evaluation process, including the criteria for judging aesthetic quality, within the <Thought> tag. Then, give a final score from 1 to 5 within the <Score> tag.
<Thought>
[Analyze the evaluation process in detail here]
</Thought>
<Score>X</Score>'''
    
    def pil_image_to_base64(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image;base64,{encoded_image_text}"
    
    def extract_scores(output_text):
        scores = []
        for text in output_text:
            match = re.search(r'<Score>(\d+)</Score>', text)
            if match:
                scores.append(float(match.group(1))/5)
            else:
                scores.append(0)
        return scores
    
    def _fn(images, prompts, metadata):
        images = convert_images_to_pil_list(images)
        
        images_base64 = [pil_image_to_base64(image) for image in images]
        messages = []
        for base64_qwen in images_base64:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_qwen},
                        {"type": "text", "text": task},
                    ],
                },
            ])
        
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        rewards = extract_scores(output_texts)
        return np.array(rewards), {}
    
    return _fn


def ocr_score(device):
    """OCR score using PaddleOCR."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("PaddleOCR not installed. Please install with: pip install paddleocr")
    
    from Levenshtein import distance
    
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        prompts_text = [prompt.split('"')[1] if '"' in prompt else prompt for prompt in prompts]
        rewards = []
        
        for img, prompt in zip(images, prompts_text):
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                result = ocr.ocr(img, cls=False)
                recognized_text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                    dist = min(dist, len(prompt))
            except Exception as e:
                dist = len(prompt)
            
            reward = 1 - dist / len(prompt) if len(prompt) > 0 else 0
            rewards.append(reward)
        
        return np.array(rewards), {}
    
    return _fn


def video_ocr_score(device):
    """Video OCR score using PaddleOCR."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("PaddleOCR not installed")
    
    from Levenshtein import distance
    
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
    frame_interval = 4
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        
        prompts_text = [prompt.split('"')[1] if '"' in prompt else prompt for prompt in prompts]
        rewards = []
        
        for img, prompt in zip(images, prompts_text):
            prompt = prompt.replace(' ', '').lower()
            frame_rewards = []
            
            # Handle video
            if isinstance(img, np.ndarray) and img.ndim == 4:
                sampled_frames = img[::frame_interval]
            else:
                sampled_frames = [img]
            
            for frame in sampled_frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                try:
                    result = ocr.ocr(frame, cls=False)
                    text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                    text = text.replace(' ', '').lower()
                    
                    dist = distance(text, prompt)
                    dist = min(dist, len(prompt))
                except Exception:
                    dist = len(prompt)
                
                reward = 1 - dist / len(prompt) if len(prompt) > 0 else 0
                if reward > 0:
                    frame_rewards.append(reward)
            
            if frame_rewards:
                rewards.append(sum(frame_rewards) / len(frame_rewards))
            else:
                rewards.append(0.0)
        
        return np.array(rewards), {}
    
    return _fn


def deqa_score_remote(device):
    """DeQA remote score."""
    import requests
    from requests.adapters import HTTPAdapter, Retry
    import pickle
    
    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        
        for image_batch in images_batched:
            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())
            
            data = {"images": jpeg_images}
            data_bytes = pickle.dumps(data)
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)
            all_scores += response_data["outputs"]
        
        return np.array(all_scores), {}
    
    return _fn


def geneval_score(device):
    """GenEval remote score."""
    import requests
    from requests.adapters import HTTPAdapter, Retry
    import pickle
    
    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    
    def _fn(images, prompts, metadatas, only_strict):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())
            
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)
            
            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        
        # Merge group rewards
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        
        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        
        return all_scores, all_rewards, all_strict_rewards, dict(all_group_rewards_dict), dict(all_group_strict_rewards_dict)
    
    return _fn


def unifiedreward_score_remote(device):
    """UnifiedReward remote score."""
    import requests
    from requests.adapters import HTTPAdapter, Retry
    import pickle
    
    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))
        
        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())
            
            data = {"images": jpeg_images, "prompts": prompt_batch}
            data_bytes = pickle.dumps(data)
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)
            all_scores += response_data["outputs"]
        
        return np.array(all_scores), {}
    
    return _fn


def unifiedreward_score_sglang(device):
    """UnifiedReward score using SGLang."""
    import asyncio
    from openai import AsyncOpenAI
    import base64
    import re
    
    def pil_image_to_base64(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image;base64,{encoded_image_text}"
    
    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores
    
    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
    
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": images_base64}},
                    {"type": "text", "text": question},
                ],
            }],
            temperature=0,
        )
        return response.choices[0].message.content
    
    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results
    
    def _fn(images, prompts, metadata):
        images = convert_images_to_pil_list(images)
        images = [image.resize((512, 512)) for image in images]
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return np.array(score), {}
    
    return _fn


def altclip_score(device, ft_model_path=None, pretrained_model_name_or_path=None, processor_cache_dir=None, resize_res=512):
    """AltCLIP score for text-image alignment.
    Supports both images and videos. For videos, extracts frames at 1 fps and averages frame scores.
    
    Args:
        device: Device to run the model on
        ft_model_path: Path to fine-tuned AltCLIP model checkpoint. If None, uses default from REWARD_MODEL_PATH.
        pretrained_model_name_or_path: Path to pretrained AltCLIP model for processor. If None, uses default.
        processor_cache_dir: Cache directory for processor. If None, uses default.
        resize_res: Image resize resolution. Default is 512.
    """
    from hymm.models.reward_models.altclip_rm import AltCLIPRM
    
    # Use defaults from REWARD_MODEL_PATH if not provided
    if ft_model_path is None:
        # Default fine-tuned model path - users should override if needed
        # ft_model_path = REWARD_MODEL_PATH["altclip_ft_017"]
        ft_model_path = REWARD_MODEL_PATH["altclip_ft_024"]
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = REWARD_MODEL_PATH["altclip"]
    if processor_cache_dir is None:
        processor_cache_dir = REWARD_MODEL_PATH["altclip"]
    
    # Initialize the model
    model = AltCLIPRM(
        ft_model_path=ft_model_path,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        resize_res=resize_res,
        processor_cache_dir=processor_cache_dir,
    )
    model.eval()
    model.to(device)
    
    def _fn(images_or_video_paths, prompts, metadata):
        """
        Args:
            images_or_video_paths: Either:
                - torch.Tensor (NCHW) or list of PIL Images/numpy arrays for images
                - List[str] of video file paths for videos
            prompts: List of text prompts
            metadata: Optional metadata (not used)
        
        Returns:
            scores: numpy array of scores (one per prompt-image/video pair)
            meta_dict: Empty dict
        """
        # Check if input is video paths (list of strings) or images (tensor/PIL/etc)
        is_video_input = _is_video_input(images_or_video_paths)
        
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if is_video_input:
            # Process videos: extract frames and compute average scores
            video_paths = images_or_video_paths
            video_scores = []
            
            for video_path, prompt in zip(video_paths, prompts):
                try:
                    # Extract frames from video at 1 fps (1 frame per second)
                    # This provides good coverage while keeping computational cost reasonable
                    frames = _extract_frames_from_video(video_path, target_fps=2.0)
                    
                    if frames.shape[0] == 0:
                        # If no frames extracted, return NaN
                        video_scores.append(float('nan'))
                        continue
                    
                    # Convert frames tensor (T, C, H, W) to PIL Images
                    frame_pil_images = _convert_frames_to_pil_images(frames)
                    
                    # Compute scores for all frames with the same prompt
                    frame_prompts = [prompt] * len(frame_pil_images)
                    
                    with torch.no_grad():
                        rm_scores, probs = model.get_preference_scores(frame_prompts, frame_pil_images)
                    
                    # Extract diagonal elements for matching prompt[i] with frame[i]
                    num_frames = len(frame_pil_images)
                    similarity_matrix = rm_scores.view(num_frames, num_frames)
                    frame_scores = torch.diagonal(similarity_matrix)
                    
                    # Average frame scores to get video score
                    video_score = frame_scores.cpu().float().mean().item()
                    video_scores.append(video_score)
                
                except Exception as e:
                    # On error, return NaN for this video
                    import traceback
                    print(f"Warning: Error processing video {video_path}: {e}")
                    traceback.print_exc()
                    video_scores.append(float('nan'))
            
            return np.array(video_scores), {}
        
        else:
            # Process images (original logic)
            images = convert_images_to_pil_list(images_or_video_paths)
            
            # Get scores using get_preference_scores
            # This returns (rm_scores, probs) where rm_scores is flattened similarity matrix
            # In reward function context, prompts and images should always be matched one-to-one
            assert len(prompts) == len(images), f"Prompts and images must have the same length, got {len(prompts)} prompts and {len(images)} images"
            
            with torch.no_grad():
                rm_scores, probs = model.get_preference_scores(prompts, images)
            # Extract diagonal elements for matching prompt[i] with image[i]
            # Reshape flattened similarity matrix (N*N) to (N, N) and take diagonal
            num_pairs = len(prompts)
            similarity_matrix = rm_scores.view(num_pairs, num_pairs)
            scores = torch.diagonal(similarity_matrix)
            
            # Convert to float32 before converting to numpy (bfloat16 is not supported by numpy)
            return scores.cpu().float().numpy(), {}
    
    # Store model reference for offloading
    _fn._reward_model = model
    return _fn


def hpsv3_score(device):
    """HPSv3 score for image quality and text alignment.
    Supports both images and videos. For videos, extracts frames at video fps and averages frame scores.
    
    Args:
        device: Device to run the model on
        checkpoint_path: Optional path to HPSv3 checkpoint. If None, downloads from HuggingFace.
        model_name_or_path: Optional path to base model (Qwen2-VL). If None, uses value from config.
    """
    from hymm.models.reward_models.hpsv3 import HPSv3RewardInferencer
    import tempfile
    import os
    
    # Initialize the inferencer
    inferencer = HPSv3RewardInferencer(
        checkpoint_path=REWARD_MODEL_PATH["hpsv3"],
        device=device,
        model_name_or_path=REWARD_MODEL_PATH["qwen2_vl_7b"]
    )
    
    def _fn(images_or_video_paths, prompts, metadata):
        """
        Args:
            images_or_video_paths: Either:
                - torch.Tensor (NCHW) or list of PIL Images/numpy arrays for images
                - List[str] of video file paths for videos
            prompts: List of text prompts
            metadata: Optional metadata (not used)
        
        Returns:
            scores: numpy array of scores
            meta_dict: Empty dict
        """
        # Check if input is video paths (list of strings) or images (tensor/PIL/etc)
        is_video_input = _is_video_input(images_or_video_paths)
        
        if is_video_input:
            # Process videos: extract frames and compute average scores
            video_paths = images_or_video_paths
            video_scores = []
            
            for video_path, prompt in zip(video_paths, prompts):
                try:
                    # Extract frames from video at 1 fps (1 frame per second)
                    # This provides good coverage while keeping computational cost reasonable
                    frames = _extract_frames_from_video(video_path, target_fps=4.0)
                    if frames.shape[0] == 0:
                        # If no frames extracted, return NaN
                        video_scores.append(float('nan'))
                        continue
                    
                    # Convert frames tensor (T, C, H, W) to PIL Images
                    frame_pil_images = _convert_frames_to_pil_images(frames)
                    
                    # Save frames to temporary files and batch process
                    with _TempFileManager() as temp_mgr:
                        frame_paths = []
                        for frame_pil in frame_pil_images:
                            temp_path = temp_mgr.create_temp_file(suffix='.png')
                            frame_pil.save(temp_path, format='PNG')
                            frame_paths.append(temp_path)
                        
                        # Batch process all frames with the same prompt
                        frame_prompts = [prompt] * len(frame_paths)
                        frame_rewards = inferencer.reward(frame_prompts, image_paths=frame_paths)
                        # Extract scores (mu values, first element of each reward tensor)
                        frame_scores = [reward[0].item() for reward in frame_rewards]
                        
                        # Average frame scores to get video score
                        video_score = np.mean(frame_scores)
                        video_scores.append(video_score)
                
                except Exception as e:
                    # On error, return NaN for this video
                    import traceback
                    print(f"Warning: Error processing video {video_path}: {e}")
                    traceback.print_exc()
                    video_scores.append(float('nan'))
            
            return np.array(video_scores), {}
        
        else:
            # Process images (original logic)
            images = convert_images_to_pil_list(images_or_video_paths)
            
            # Save PIL images to temporary files (HPSv3 expects file paths)
            with _TempFileManager() as temp_mgr:
                image_paths = []
                for pil_img in images:
                    temp_path = temp_mgr.create_temp_file(suffix='.png')
                    pil_img.save(temp_path, format='PNG')
                    image_paths.append(temp_path)
                
                # Get rewards from HPSv3
                rewards = inferencer.reward(prompts, image_paths=image_paths)
                
                # Extract scores (mu values, first element of each reward tensor)
                scores = [reward[0].item() for reward in rewards]
                
                return np.array(scores), {}
    
    # Store inferencer reference for offloading
    _fn._reward_model = inferencer
    return _fn


def hpsv3_general_score(device):
    """HPSv3-general score for visual quality assessment only.
    Uses fixed prompt "A high-quality image" to focus exclusively on visual quality.
    For videos, extracts all frames at 24fps and computes mean score across all frames.
    
    Based on the paper description:
    - Uses general prompt "A high-quality image" instead of user prompt
    - Computes mean score of all frames (at 24fps) for comprehensive quality assessment
    
    Args:
        device: Device to run the model on
    """
    from hymm.models.reward_models.hpsv3 import HPSv3RewardInferencer
    import tempfile
    import os
    
    # Initialize the inferencer
    inferencer = HPSv3RewardInferencer(
        checkpoint_path=REWARD_MODEL_PATH["hpsv3"],
        device=device,
        model_name_or_path=REWARD_MODEL_PATH["qwen2_vl_7b"]
    )
    
    # Fixed prompt for visual quality assessment
    QUALITY_PROMPT = "A high-quality image."
    
    def _fn(images_or_video_paths, prompts, metadata):
        """
        Args:
            images_or_video_paths: Either:
                - torch.Tensor (NCHW) or list of PIL Images/numpy arrays for images
                - List[str] of video file paths for videos
            prompts: List of text prompts (ignored, using fixed quality prompt)
            metadata: Optional metadata (not used)
        
        Returns:
            scores: numpy array of scores
            meta_dict: Empty dict
        """
        # Check if input is video paths (list of strings) or images (tensor/PIL/etc)
        is_video_input = _is_video_input(images_or_video_paths)
        
        if is_video_input:
            # Process videos: extract frames at 24fps and compute mean scores across ALL frames
            video_paths = images_or_video_paths
            video_scores = []
            
            for video_path in video_paths:
                try:
                    # Extract frames from video at 24 fps to get all frames
                    # This provides comprehensive coverage of visual quality across entire video
                    frames = _extract_frames_from_video(video_path, target_fps=24.0)
                    if frames.shape[0] == 0:
                        # If no frames extracted, return NaN
                        video_scores.append(float('nan'))
                        continue
                    
                    # Convert frames tensor (T, C, H, W) to PIL Images
                    frame_pil_images = _convert_frames_to_pil_images(frames)
                    
                    # Save frames to temporary files and batch process
                    with _TempFileManager() as temp_mgr:
                        frame_paths = []
                        for frame_pil in frame_pil_images:
                            temp_path = temp_mgr.create_temp_file(suffix='.png')
                            frame_pil.save(temp_path, format='PNG')
                            frame_paths.append(temp_path)
                        
                        # Batch process all frames with fixed quality prompt
                        frame_prompts = [QUALITY_PROMPT] * len(frame_paths)
                        frame_rewards = inferencer.reward(frame_prompts, image_paths=frame_paths)
                        # Extract scores (mu values, first element of each reward tensor)
                        frame_scores = [reward[0].item() for reward in frame_rewards]
                        
                        # Compute mean of ALL frame scores for comprehensive quality assessment
                        video_score = np.mean(frame_scores)
                        video_scores.append(video_score)
                
                except Exception as e:
                    # On error, return NaN for this video
                    import traceback
                    print(f"Warning: Error processing video {video_path}: {e}")
                    traceback.print_exc()
                    video_scores.append(float('nan'))
            
            return np.array(video_scores), {}
        
        else:
            # Process images with fixed quality prompt
            images = convert_images_to_pil_list(images_or_video_paths)
            
            # Save PIL images to temporary files (HPSv3 expects file paths)
            with _TempFileManager() as temp_mgr:
                image_paths = []
                for pil_img in images:
                    temp_path = temp_mgr.create_temp_file(suffix='.png')
                    pil_img.save(temp_path, format='PNG')
                    image_paths.append(temp_path)
                
                # Get rewards from HPSv3 using fixed quality prompt
                quality_prompts = [QUALITY_PROMPT] * len(image_paths)
                rewards = inferencer.reward(quality_prompts, image_paths=image_paths)
                
                # Extract scores (mu values, first element of each reward tensor)
                scores = [reward[0].item() for reward in rewards]
                
                return np.array(scores), {}
    
    # Store inferencer reference for offloading
    _fn._reward_model = inferencer
    return _fn


def hpsv3_remote_score(device, server_url=None, mode="random"):
    """HPSv3 remote score for image quality and text alignment.
    Supports both images and videos. For videos, extracts frames and averages frame scores.
    
    Args:
        device: Device to run models on (not used for remote service, kept for compatibility)
        server_url: The request URL(s) for the inference service. Can be:
                    - Single URL: "http://host:port" or "host:port"
                    - Multiple URLs (comma/semicolon/space separated): "host1:port1,host2:port2"
                    - List of URLs: ["http://host1:port1", "http://host2:port2"]
                    If None, raises ValueError.
        mode: Load balancing mode. Either "random" (default) or "round_robin".
              - "random": Randomly select a server for each request
              - "round_robin": Rotate through servers in order
    
    Returns:
        A function that takes (images_or_video_paths, prompts, metadata) and returns (scores, meta_dict)
    """
    import requests
    import base64
    import random
    from requests.adapters import HTTPAdapter, Retry
    
    if server_url is None:
        raise ValueError("server_url is required for hpsv3_remote_score. Please provide it in the format 'http://host:port' or 'host1:port1,host2:port2'")
    
    # Parse server_url: support string (single or comma-separated) or list
    if isinstance(server_url, str):
        # Normalize separators and split
        normalized = (
            server_url
            .replace("\n", ",")
            .replace("\t", ",")
            .replace(";", ",")
            .replace(" ", ",")
        )
        server_urls = [p.strip() for p in normalized.split(",") if p.strip()]
    elif isinstance(server_url, (list, tuple)):
        server_urls = list(server_url)
    else:
        raise ValueError(f"server_url must be a string or list, got {type(server_url)}")
    
    if not server_urls:
        raise ValueError("At least one server_url is required")
    
    # Normalize each server_url: add http:// prefix if missing
    normalized_server_urls = []
    for url in server_urls:
        if not url.startswith(('http://', 'https://')):
            url = f"http://{url}"
        # Normalize server_url (ensure it doesn't end with /)
        url = url.rstrip('/')
        normalized_server_urls.append(url)
    
    server_urls = normalized_server_urls
    
    # Validate mode
    if mode not in ["random", "round_robin"]:
        raise ValueError(f"mode must be 'random' or 'round_robin', got '{mode}'")
    
    # Create sessions with retry logic for each server
    sessions = []
    for url in server_urls:
        sess = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        sess.mount("http://", HTTPAdapter(max_retries=retries))
        sessions.append(sess)
    
    # Round-robin counter (thread-safe with local variable)
    round_robin_counter = [0]  # Use list to allow modification in nested function
    
    def pil_img2b64(img_pil):
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def select_server_index():
        """Select a server index based on mode."""
        if mode == "random":
            return random.randint(0, len(server_urls) - 1)
        elif mode == "round_robin":
            idx = round_robin_counter[0] % len(server_urls)
            round_robin_counter[0] += 1
            return idx
    
    def make_request_with_failover(data, timeout=120):
        """Make request with failover support. Tries all servers in order if one fails."""
        last_error = None
        # Try servers in order (starting from selected index for round_robin, or random for random mode)
        start_idx = select_server_index()
        server_indices = list(range(start_idx, len(server_urls))) + list(range(0, start_idx))
        for idx in server_indices:
            try:
                response = sessions[idx].post(
                    f"{server_urls[idx]}/",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
                if response.status_code == 200:
                    return response, None
                else:
                    last_error = f"Server {server_urls[idx]} returned status code {response.status_code}"
            except Exception as e:
                last_error = f"Server {server_urls[idx]} error: {str(e)}"
                continue
        
        # All servers failed
        return None, last_error
    
    def _fn(images_or_video_paths, prompts, metadata):
        """
        Args:
            images_or_video_paths: Either:
                - torch.Tensor (NCHW) or list of PIL Images/numpy arrays for images
                - List[str] of video file paths for videos
            prompts: List of text prompts
            metadata: Optional metadata (not used)
        
        Returns:
            scores: numpy array of scores (one per prompt-image/video pair)
            meta_dict: Empty dict
        """
        # Check if input is video paths (list of strings) or images (tensor/PIL/etc)
        is_video_input = _is_video_input(images_or_video_paths)
        
        if is_video_input:
            # Process videos: extract frames and compute average scores
            video_paths = images_or_video_paths
            video_scores = []
            
            for video_path, prompt in zip(video_paths, prompts):
                try:
                    # Extract frames from video at 4 fps (similar to local hpsv3_score)
                    frames = _extract_frames_from_video(video_path, target_fps=1.0)
                    if frames.shape[0] == 0:
                        # If no frames extracted, return NaN
                        video_scores.append(float('nan'))
                        continue
                    
                    # Convert frames tensor (T, C, H, W) to PIL Images
                    frame_pil_images = _convert_frames_to_pil_images(frames)
                    
                    # Convert frames to base64
                    frame_base64_list = [pil_img2b64(frame_pil) for frame_pil in frame_pil_images]
                    
                    # Prepare data for remote request
                    # For video frames, we send each frame as a separate data item with the same prompt
                    data = [
                        {"img_base64": img_base64, "prompt": prompt, "data_type": "IMAGE"}
                        for img_base64 in frame_base64_list
                    ]
                    
                    # Prepare request data
                    request_data = {
                        "data_size": len(data),
                        "data": data,
                        "stats": ["hpsv3_server"],
                        "stat_config": {"hpsv3_server": {}},
                        "request_id": "hpsv3_remote_request",
                        "request_ts": 0,
                        "source": "hpsv3_remote",
                        "bid": "BID_hpsv3_remote",
                        "principal": "hpsv3_remote",
                    }
                    
                    # Make remote request with failover
                    response, error = make_request_with_failover(request_data, timeout=120)
                    
                    # Parse response
                    if response is None:
                        print(f"Warning: All HPSv3 servers failed. Last error: {error}")
                        video_scores.append(float('nan'))
                        continue
                    
                    try:
                        response_data = response.json()
                        if response_data.get("code") != 200:
                            print(f"Warning: Remote HPSv3 service returned error: {response_data.get('message', 'Unknown error')}")
                            video_scores.append(float('nan'))
                            continue
                        
                        # Extract scores from response
                        frame_scores = []
                        for item in response_data.get("data", []):
                            hpsv3_result = item.get("hpsv3", {})
                            if hpsv3_result.get("code") == 200:
                                value = hpsv3_result.get("value")
                                if value is not None:
                                    frame_scores.append(float(value))
                        
                        if len(frame_scores) > 0:
                            # Average frame scores to get video score
                            video_score = np.mean(frame_scores)
                            video_scores.append(video_score)
                        else:
                            print(f"Warning: No valid scores returned for video {video_path}")
                            video_scores.append(float('nan'))
                    
                    except Exception as e:
                        print(f"Warning: Error parsing response for video {video_path}: {e}")
                        video_scores.append(float('nan'))
                
                except Exception as e:
                    # On error, return NaN for this video
                    import traceback
                    print(f"Warning: Error processing video {video_path}: {e}")
                    traceback.print_exc()
                    video_scores.append(float('nan'))
            
            return np.array(video_scores), {}
        
        else:
            # Process images
            images = convert_images_to_pil_list(images_or_video_paths)
            
            # Convert images to base64
            image_base64_list = [pil_img2b64(img) for img in images]
            
            # Prepare data for remote request
            data = [
                {"img_base64": img_base64, "prompt": prompt, "data_type": "IMAGE"}
                for img_base64, prompt in zip(image_base64_list, prompts)
            ]
            
            # Prepare request data
            request_data = {
                "data_size": len(data),
                "data": data,
                "stats": ["hpsv3_server"],
                "stat_config": {"hpsv3_server": {}},
                "request_id": "hpsv3_remote_request",
                "request_ts": 0,
                "source": "hpsv3_remote",
                "bid": "BID_hpsv3_remote",
                "principal": "hpsv3_remote",
            }
            
            # Make remote request with failover
            try:
                response, error = make_request_with_failover(request_data, timeout=120)
                
                # Parse response
                if response is None:
                    print(f"Warning: All HPSv3 servers failed. Last error: {error}")
                    return np.array([float('nan')] * len(prompts)), {}
                
                response_data = response.json()
                if response_data.get("code") != 200:
                    print(f"Warning: Remote HPSv3 service returned error: {response_data.get('message', 'Unknown error')}")
                    return np.array([float('nan')] * len(prompts)), {}
                
                # Extract scores from response
                scores = []
                for item in response_data.get("data", []):
                    hpsv3_result = item.get("hpsv3", {})
                    if hpsv3_result.get("code") == 200:
                        value = hpsv3_result.get("value")
                        if value is not None:
                            scores.append(float(value))
                        else:
                            scores.append(float('nan'))
                    else:
                        scores.append(float('nan'))
                
                # Ensure we have scores for all inputs
                while len(scores) < len(prompts):
                    scores.append(float('nan'))
                
                return np.array(scores), {}
            
            except Exception as e:
                import traceback
                print(f"Warning: Error calling remote HPSv3 service: {e}")
                traceback.print_exc()
                return np.array([float('nan')] * len(prompts)), {}
    
    return _fn


def _extract_frames_from_video(video_path, target_fps=1.0):
    """
    Extract frames from video at specified fps (default 1.0 fps for reasonable frame count).
    
    Args:
        video_path: Path to video file
        target_fps: Target fps for frame extraction. Default is 1.0 fps (1 frame per second).
                    This provides a good balance between coverage and computational cost.
    
    Returns:
        torch.Tensor of frames with shape (T, C, H, W)
    """
    try:
        import decord
        # Set bridge to torch to avoid mxnet dependency issues
        # This allows decord to work with torch tensors directly
        try:
            decord.bridge.set_bridge('torch')
        except Exception:
            # If torch bridge is not available, try numpy
            try:
                decord.bridge.set_bridge('numpy')
            except Exception:
                # If both fail, continue with default (mxnet)
                pass
    except ImportError:
        raise ImportError("decord is required for video frame extraction. Install with: pip install decord")
    
    # Check if file exists
    import os
    if not os.path.exists(video_path):
        print(f"Warning: Video file does not exist: {video_path}")
        return torch.empty((0, 3, 0, 0))
    
    # Read video
    try:
        vr = decord.VideoReader(video_path)
    except Exception as e:
        print(f"Warning: Failed to open video with decord: {video_path}, error: {e}")
        return torch.empty((0, 3, 0, 0))
    
    total_frames = len(vr)
    try:
        video_fps = vr.get_avg_fps()
    except Exception:
        # If fps cannot be read, use a default value
        video_fps = 24.0
        print(f"Warning: Could not read fps from video {video_path}, using default 24.0 fps")
    
    if total_frames == 0:
        print(f"Warning: Video has 0 frames: {video_path}")
        return torch.empty((0, 3, 0, 0))
    
    # Calculate frame interval based on target_fps
    # If video_fps is 24 and target_fps is 1.0, we extract every 24th frame
    frame_interval = max(1, int(round(video_fps / target_fps)))
    frame_indices = list(range(0, total_frames, frame_interval))
    
    # Ensure we get at least one frame
    if len(frame_indices) == 0:
        frame_indices = [0]
        
    # Extract frames
    try:
        frame_batch = vr.get_batch(frame_indices)
        
        # Handle different bridge backends
        if hasattr(frame_batch, 'asnumpy'):
            # mxnet backend
            frames_np = frame_batch.asnumpy()  # (T, H, W, C)
        elif isinstance(frame_batch, torch.Tensor):
            # torch backend - already a tensor
            frames_tensor = frame_batch.permute(0, 3, 1, 2).float() / 255.0
            return frames_tensor
        else:
            # numpy backend or other
            frames_np = np.array(frame_batch)
        
        # Convert to (T, C, H, W) tensor and normalize to [0, 1]
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        return frames_tensor
    except Exception as e:
        print(f"Warning: Batch extraction failed for {video_path}, error: {e}, trying frame-by-frame...")
        # Fallback: try to get frames one by one
        frames = []
        for idx in frame_indices:
            try:
                frame_batch = vr.get_batch([idx])
                
                # Handle different bridge backends
                if hasattr(frame_batch, 'asnumpy'):
                    frame_np = frame_batch.asnumpy()[0]  # (H, W, C)
                elif isinstance(frame_batch, torch.Tensor):
                    frame_tensor = frame_batch[0].permute(2, 0, 1).float() / 255.0
                    frames.append(frame_tensor)
                    continue
                else:
                    frame_np = np.array(frame_batch)[0]
                
                frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
                frames.append(frame_tensor)
            except Exception as frame_error:
                print(f"Warning: Failed to extract frame {idx}: {frame_error}")
                continue
        if len(frames) == 0:
            print(f"Warning: Failed to extract any frames from {video_path}")
            return torch.empty((0, 3, 0, 0))
        print(f"Successfully extracted {len(frames)} frames frame-by-frame")
        return torch.stack(frames, dim=0)
        
def _hyvideoreward_remote_score_impl(device, mode="random", server_url=None, metric_weights=None):
    """
    HyVideoReward remote reward service (internal implementation).
    
    Args:
        device: Device to run models on (not used for remote service, kept for compatibility)
        mode: Mode for selecting server, "random" or "min"
        server_url: The request URL for the inference service
        metric_weights: Dict of metric_name -> weight. If provided, only metrics with non-zero weights will be returned.
                       Example: {"VQ": 0.0, "MQ": 1.0, "TA": 0.0, "AES": 0.0}
    """
    from hymm.models.reward_models.hyvideo_remote_reward import HyVideoRewardRemote, DEFAULT_SERVER_URL 

    if server_url is None:
        server_url = DEFAULT_SERVER_URL
    
    # Initialize the inferencer
    inferencer = HyVideoRewardRemote(
        server_url=server_url,
        mode=mode
    )
    
    # Determine which metrics to return based on metric_weights
    if metric_weights is not None and isinstance(metric_weights, dict):
        # Only return metrics with non-zero weights
        metrics_to_return = {k for k, v in metric_weights.items() if v != 0.0}
    else:
        metrics_to_return = None  # Return all metrics
    
    def _fn(video_paths: List[str], prompts: List[str], metadata: List[Dict] = None):
        if metadata is None:
            metadata = [{}] * len(video_paths)

        # Send the whole batch in ONE dispatcher request; the dispatcher fans it out
        # to the healthy replicas and does the load balancing server-side.
        all_entries = inferencer.reward(video_paths, prompts)
        #Convert entries to scores_dict using common utility
        scores_dict = _convert_reward_entries_to_scores_dict(all_entries, metrics_to_return=metrics_to_return)
         
        return scores_dict, {}
    
    return _fn

# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_key(key: str) -> str:
    """Normalize key to lowercase for case-insensitive lookup."""
    return str(key).strip().lower()


def _convert_reward_entries_to_scores_dict(
    all_entries: List[Dict[str, Any]], 
    metrics_to_return: set = None
) -> Dict[str, List[float]]:
    """
    Convert a list of reward entry dictionaries to scores_dict format.
    
    Args:
        all_entries: List of dictionaries, each containing reward metrics
        metrics_to_return: Optional set of metric keys to filter. If None, returns all metrics.
    
    Returns:
        scores_dict: Dict of {metric_name: List[float]} where each metric corresponds to a dimension
    """
    scores_dict = {}
    
    if not all_entries:
        return scores_dict
    
    # Get all possible keys from entries
    all_keys = set()
    for entry in all_entries:
        all_keys.update(entry.keys())
    
    # Filter keys based on metrics_to_return if provided
    if metrics_to_return is not None:
        all_keys = {k for k in all_keys if k in metrics_to_return}
    
    # Extract each metric
    for key in all_keys:
        scores = []
        for entry in all_entries:
            val = entry.get(key)
            try:
                scores.append(float(val) if val is not None else float('nan'))
            except (TypeError, ValueError):
                scores.append(float('nan'))
        scores_dict[key] = scores
    
    return scores_dict


def _convert_frames_to_pil_images(frames: torch.Tensor) -> List[Image.Image]:
    """
    Convert video frames tensor (T, C, H, W) to a list of PIL Images.
    
    Args:
        frames: Tensor of shape (T, C, H, W) with values in [0, 1]
    
    Returns:
        List of PIL.Image objects
    """
    frame_pil_images = []
    num_frames = frames.shape[0]
    
    for t in range(num_frames):
        frame = frames[t]  # (C, H, W)
        # Convert tensor to numpy array (H, W, C)
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        
        # Normalize to [0, 255] if needed (frames are already in [0, 1])
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
        
        frame_pil = Image.fromarray(frame_np)
        frame_pil_images.append(frame_pil)
    
    return frame_pil_images


def _is_video_input(images_or_video_paths: Union[torch.Tensor, List, str]) -> bool:
    """
    Check if input is video paths (list of strings) or images.
    
    Args:
        images_or_video_paths: Either video paths (list of strings) or images
    
    Returns:
        True if input is video paths, False otherwise
    """
    return isinstance(images_or_video_paths, list) and len(images_or_video_paths) > 0 and isinstance(images_or_video_paths[0], str)


class _TempFileManager:
    """Context manager for managing temporary files."""
    
    def __init__(self):
        self.temp_files = []
    
    def create_temp_file(self, suffix='.png', delete=False):
        """Create a temporary file and track it."""
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=delete, suffix=suffix)
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up all temporary files."""
        import os
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        return False


def convert_images_to_pil_list(images: Union[torch.Tensor, np.ndarray, List, Image.Image, str]) -> List[Image.Image]:
    """
    Convert various image formats to a list of PIL Images.
    
    Supports:
    - torch.Tensor: (N, C, H, W) or (C, H, W) format, values in [0, 1] or [0, 255]
    - np.ndarray: (N, H, W, C), (N, C, H, W), (H, W, C), or (C, H, W) format
    - List: List of torch.Tensor, np.ndarray, PIL.Image, or file paths (str)
    - PIL.Image: Single image
    - str: Single file path
    
    Args:
        images: Images in various formats
        
    Returns:
        List of PIL.Image objects
    """
    if isinstance(images, torch.Tensor):
        # Handle batch tensor (NCHW format)
        if images.dim() == 4:
            # Batch of images: (N, C, H, W)
            images_list = []
            for i in range(images.shape[0]):
                img = images[i]  # (C, H, W)
                if img.shape[0] == 3:  # CHW format
                    img = img.permute(1, 2, 0)  # CHW -> HWC
                # Convert to uint8 if needed
                if img.max() <= 1.0:
                    img = (img * 255).round().clamp(0, 255)
                img = img.to(torch.uint8).cpu().numpy()
                images_list.append(Image.fromarray(img))
            return images_list
        elif images.dim() == 3:
            # Single image: (C, H, W)
            if images.shape[0] == 3:  # CHW format
                images = images.permute(1, 2, 0)  # CHW -> HWC
            # Convert to uint8 if needed
            if images.max() <= 1.0:
                images = (images * 255).round().clamp(0, 255)
            images = images.to(torch.uint8).cpu().numpy()
            return [Image.fromarray(images)]
        else:
            raise ValueError(f"Unsupported tensor shape: {images.shape}")
    
    elif isinstance(images, np.ndarray):
        # Handle numpy array
        if images.ndim == 4:
            # Batch: (N, H, W, C) or (N, C, H, W)
            images_list = []
            for i in range(images.shape[0]):
                img = images[i]
                if img.shape[0] == 3:  # CHW format
                    img = img.transpose(1, 2, 0)  # CHW -> HWC
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                images_list.append(Image.fromarray(img))
            return images_list
        elif images.ndim == 3:
            # Single image
            if images.shape[0] == 3:  # CHW format
                images = images.transpose(1, 2, 0)  # CHW -> HWC
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            return [Image.fromarray(images)]
        else:
            raise ValueError(f"Unsupported numpy array shape: {images.shape}")
    
    elif isinstance(images, (list, tuple)):
        # Already a list, convert each element to PIL Image
        pil_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    if img.shape[0] == 3:  # CHW format
                        img = img.permute(1, 2, 0)  # CHW -> HWC
                    # Convert to uint8 if needed
                    if img.max() <= 1.0:
                        img = (img * 255).round().clamp(0, 255)
                    img = img.to(torch.uint8).cpu().numpy()
                    pil_images.append(Image.fromarray(img))
                else:
                    raise ValueError(f"Unsupported tensor shape in list: {img.shape}")
            elif isinstance(img, np.ndarray):
                if img.ndim == 3:
                    if img.shape[0] == 3:  # CHW format
                        img = img.transpose(1, 2, 0)  # CHW -> HWC
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(img))
                else:
                    raise ValueError(f"Unsupported numpy array shape in list: {img.shape}")
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            elif isinstance(img, str):
                # Image path
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type in list: {type(img)}")
        return pil_images
    
    elif isinstance(images, Image.Image):
        # Single PIL Image
        return [images]
    
    elif isinstance(images, str):
        # Single file path
        return [Image.open(images).convert("RGB")]
    
    else:
        raise ValueError(f"Unsupported images type: {type(images)}")


def get_image_reward_fn(reward_name: str, device, **kwargs) -> Callable:
    """
    Get an image reward function.
    
    Args:
        reward_name: Name of the reward function
        device: Device to run on
        **kwargs: Additional arguments
        
    Returns:
        Reward function
    """
    reward_functions = {
        "jpeg_incompressibility": jpeg_incompressibility,
        "jpeg_compressibility": jpeg_compressibility,
        "aesthetic": aesthetic_score,
        "clipscore": clip_score,
        "clip_score": clip_score,
        "image_similarity": image_similarity_score,
        "pickscore": pickscore_score,
        "imagereward": imagereward_score,
        "qwenvl": qwenvl_score,
        "hpsv3": hpsv3_score,
        "hpsv3_general": hpsv3_general_score,
        "hpsv3_remote": hpsv3_remote_score,
        "altclip": altclip_score,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "deqa": deqa_score_remote,
        "geneval": geneval_score,
        "unifiedreward_remote": unifiedreward_score_remote,
        "unifiedreward": unifiedreward_score_sglang,
        "unifiedreward_sglang": unifiedreward_score_sglang,
    }
    
    if reward_name not in reward_functions:
        available = ", ".join(reward_functions.keys())
        raise ValueError(f"Unknown reward: {reward_name}. Available: {available}")
    
    reward_fn_factory = reward_functions[reward_name]
    
    # Check if function requires device parameter
    import inspect
    sig = inspect.signature(reward_fn_factory)
    
    # Special handling for functions that require additional parameters
    if reward_name in ["hpsv3_remote"]:
        server_url = kwargs.get("server_url", None)
        if server_url is None:
            raise ValueError(f"server_url is required for {reward_name}. Please provide it in kwargs.")
        return reward_fn_factory(device, server_url=server_url)
    
    if 'device' in sig.parameters:
        return reward_fn_factory(device)
    else:
        return reward_fn_factory()


# ============================================================================
# Multi-Reward Interface
# ============================================================================

def multi_video_score(device, reward_config: Dict[str, Any]):
    """
    Unified multi-reward interface with flexible weighting.
    Only the **standard format** (Example 3) is supported; all other old formats are no longer supported.
    
    Args:
        device: Device to run models on
        reward_config: Dict with configuration
        
            Example:
            {
                "models": {
                    "videoalign_local": {
                        "weight": 1.0,                         # Model weight
                        "sub_reward": {"VQ": 1.0, "MQ": 1.0, "TA": 1.0},  # Sub-metric weights (optional)
                    },
                    "aesthetic": {"weight": 0.5},              # Image model, no sub-metrics
                    "clipscore": {"weight": 0.3},
                    "tencent_remote": {
                        "weight": 1.0,
                        "server_url": "http://qsave-v1-5-1119.polaris:80/inference/",  # Per-model server URL
                        "sub_reward": {"VQ": 1.0, "MQ": 1.0},
                    "hyvideo_remote": {
                        "weight": 1.0,                         # Model weight
                        "sub_reward": {"VQ": 0.0, "MQ": 1.0, "TA": 0.0, "AES": 0.0},  # Sub-metric weights (optional)
                    },
                    },
                    "video_align": {
                        "weight": 1.0,
                        "server_url": "http://server:port/",  # Each remote model has its own server_url
                        "sub_reward": {"VQ": 1.0, "MQ": 1.0, "TA": 1.0},
                    },
                },
            }
            
            Constraints:
            - Must contain a "models" field.
            - Each model's config must be a dict and must contain at least the "weight" key.
            - The optional "sub_reward" field should be a dictionary mapping metric_name to weight.
            - Remote reward models (video_align, video_score2, tencent_remote) should specify
              their own "server_url" in their model config to avoid conflicts when using multiple models.
            - If there is only one reward model, its weight must be exactly 1.0.
    
    Returns:
        reward_fn: Function (video_paths, prompts, metadata) -> (scores_dict, meta_dict)
    """
    # Must contain "models"
    if "models" not in reward_config:
        raise ValueError(
            "reward_config must contain a 'models' field with the standard format: "
            '{"models": {"model_name": {"weight": float, "sub_reward": {...}}}}'
        )
    
    model_weights = reward_config["models"]
    
    # Parse each model configuration (only standard format is supported)
    # model_configs: {model_name: {"model_weight": float, "metric_weights": dict or None}}
    model_configs: Dict[str, Dict[str, Any]] = {}
    for model_name, weight_config in model_weights.items():
        if not isinstance(weight_config, dict):
            raise ValueError(
                f"Model config for '{model_name}' must be a dict with at least 'weight' key, "
                f"got {type(weight_config)}"
            )
        
        if "weight" not in weight_config:
            raise ValueError(
                f"Model config for '{model_name}' must contain 'weight', "
                f"optionally with 'sub_reward', got keys: {list(weight_config.keys())}"
            )
        
        model_weight = float(weight_config["weight"])
        metric_weights = weight_config.get("sub_reward", None)
        
        if metric_weights is not None and not isinstance(metric_weights, dict):
            raise ValueError(
                f"'sub_reward' for model '{model_name}' must be a dict of metric_name -> weight, "
                f"got {type(metric_weights)}"
            )
        
        # Extract server_url for remote reward models (per-model configuration)
        server_url = weight_config.get("server_url", None)
        
        model_configs[model_name] = {
            "model_weight": model_weight,
            "metric_weights": metric_weights,
            "server_url": server_url,  # Per-model server_url
        }
    
    # When only one reward model is used, its weight must be 1.0
    if len(model_configs) == 1:
        only_model_name = next(iter(model_configs.keys()))
        only_model_weight = model_configs[only_model_name]["model_weight"]
        if abs(only_model_weight - 1.0) > 1e-6:
            raise ValueError(
                f"When only one reward model is used ('{only_model_name}'), "
                f"its weight must be 1.0, got {only_model_weight}"
            )
    
    # Initialize all reward functions
    reward_fns = {}
    reward_models = []  # Collect models that can be offloaded (for GPU memory management)
    for model_name in model_configs.keys():
        if model_name == "tencent_remote":
            # tencent_remote_score has default server_url, so None is acceptable
            server_url = model_configs[model_name].get("server_url")
            metric_weights = model_configs[model_name].get("metric_weights")
            if metric_weights is not None and isinstance(metric_weights, dict):
                # Use sub_reward keys as dimensions
                dimensions = list(metric_weights.keys())
            else:
                # Fallback to default dimensions if sub_reward is not provided
                dimensions = None
            reward_fns[model_name] = tencent_remote_score(server_url, dimensions=dimensions)
        elif model_name in ("hyvideoreward_remote", "hyvideoreward_remote_pre_mq", "hyvideoreward_remote_ta", "hyvideoreward_remote_vq", "hyvideoreward_remote_lrm"):
            server_url = model_configs[model_name].get("server_url")
            metric_weights = model_configs[model_name].get("metric_weights")
            reward_fns[model_name] = _hyvideoreward_remote_score_impl(device, server_url=server_url, metric_weights=metric_weights)
        elif model_name == "hpsv3_remote":
            # hpsv3_remote requires server_url
            server_url = model_configs[model_name].get("server_url")
            if server_url is None:
                raise ValueError(
                    f"server_url required for {model_name}. "
                    f"Please provide it in reward_config['models']['{model_name}']['server_url']"
                )
            # Support mode parameter for load balancing (default: "random")
            mode = model_configs[model_name].get("mode", "random")
            reward_fns[model_name] = hpsv3_remote_score(device, server_url=server_url, mode=mode)
        elif model_name == "dynamic_degree":
            # Dynamic degree using RAFT optical flow
            model_path = model_configs[model_name].get("model_path", None)
            normalize = model_configs[model_name].get("normalize", True)
            target_fps = model_configs[model_name].get("target_fps", 8.0)
            reward_fn = dynamic_degree_score(device, model_path=model_path, normalize=normalize, target_fps=target_fps)
            reward_fns[model_name] = reward_fn
            if hasattr(reward_fn, '_reward_model'):
                reward_models.append(reward_fn._reward_model)
        elif model_name == "dynamic_degree_simple":
            # Dynamic degree using simple frame difference (lightweight)
            normalize = model_configs[model_name].get("normalize", True)
            target_fps = model_configs[model_name].get("target_fps", 8.0)
            reward_fns[model_name] = dynamic_degree_score_simple(device, target_fps=target_fps, normalize=normalize)
        else:
            # Try image reward
            try:
                reward_fn = get_image_reward_fn(model_name, device)
                reward_fns[model_name] = reward_fn
                # Store model reference for offloading if available (e.g., hpsv3, altclip, etc.)
                if hasattr(reward_fn, '_reward_model'):
                    reward_models.append(reward_fn._reward_model)
            except ValueError:
                raise ValueError(f"Unknown reward model: {model_name}")
    
    def _fn(video_paths_or_images: Union[List[str], torch.Tensor], prompts: List[str], metadata: List[Dict] = None, video_paths_latent: List[str] = None, use_latent_reward: bool = False):
        """
        Compute rewards for videos/images.
        
        Args:
            video_paths_or_images: Either a list of video file paths (for video models)
                                   or a torch.Tensor of images (for image models)
            prompts: List of text prompts
            metadata: Optional metadata for each sample
            use_latent_reward: Whether to use latent reward

        Weighting logic:
        1. Each video model's metrics are first weighted by their metric weights
        2. Then the weighted sum of each model's metrics is multiplied by model weight
        3. Image models are directly multiplied by their model weight
        4. Final avg = sum of all weighted scores
        
        Returns scores_dict with keys like:
        - "videoalign_local_vq", "videoalign_local_mq", "videoalign_local_ta" (video model metrics)
        - "aesthetic" (image model)
        - "tencent_remote_vq", "tencent_remote_mq" (another video model)
        - "avg" (weighted average)
        """
        # Determine if we're processing images or videos
        is_image_input = isinstance(video_paths_or_images, torch.Tensor)
        
        if metadata is None:
            if is_image_input:
                metadata = [{}] * video_paths_or_images.shape[0]
            else:
                metadata = [{}] * len(video_paths_or_images)
        
        all_scores = {}
        all_meta = {}
        model_raw_scores = {}  # Store raw scores by model (without prefix, for weighted avg computation)

        if use_latent_reward:
            assert video_paths_latent is not None, "video_paths_latent is required when use_latent_reward is True"
            assert len(video_paths_latent) == len(video_paths_or_images), "video_paths_latent and video_paths_or_images must have the same length"

        for model_name in model_configs.keys():
            reward_fn = reward_fns[model_name]
            
            # try:
            # Check if this is a video model, image model, or both
            video_only_models = ["videoalign_local", "video_align", "video_score2", "tencent_remote", "hyvideoreward_remote", "hyvideoreward_remote_pre_mq", "hyvideoreward_remote_ta", "hyvideoreward_remote_vq", "hyvideoreward_remote_lrm", "dynamic_degree", "dynamic_degree_simple"]
            multi_modal_models = ["hpsv3", "hpsv3_remote", "altclip", 'pickscore', 'hpsv3_general']  # Models that support both images and videos
            
            # Check if this is a video model or image model
            is_video_only_model = model_name in video_only_models
            is_multi_modal_model = model_name in multi_modal_models
            
            if is_video_only_model and is_image_input:
                # Skip video-only models when we have image input
                print(f"Warning: Skipping video-only model '{model_name}' for image input")
                continue
            elif not is_video_only_model and not is_multi_modal_model and not is_image_input:
                # Skip image-only models when we have video path input
                print(f"Warning: Skipping image-only model '{model_name}' for video path input")
                continue
            
            if model_name.endswith("_lrm"):
                scores, meta = reward_fn(video_paths_latent, prompts, metadata)
            else:
                scores, meta = reward_fn(video_paths_or_images, prompts, metadata)
            
            if isinstance(scores, dict):
                # Video models return dict of metrics
                # Store raw scores WITHOUT prefix for weighted computation
                model_raw_scores[model_name] = scores
                # Store WITH prefix for output
                for key, values in scores.items():
                    prefixed_key = f"{model_name}_{_normalize_key(key)}"
                    all_scores[prefixed_key] = values
            elif isinstance(scores, (list, np.ndarray)):
                # Image models return single score
                scores_list = list(scores) if isinstance(scores, np.ndarray) else scores
                model_raw_scores[model_name] = scores_list
                all_scores[model_name] = scores_list
            else:
                raise ValueError(f"Unexpected score format from {model_name}: {type(scores)}")
            
            all_meta.update(meta)
            
        # Return scores with prefixes (no additional normalization to avoid duplicate prefixes)
        output_scores = all_scores.copy()
        
        # Compute weighted average
        # Formula: avg = Σ(model_weight * Σ(metric_weight * metric_score))
        weighted_scores_list = []
        
        for model_name, config in model_configs.items():
            model_weight = config["model_weight"]
            per_model_metric_weights = config["metric_weights"]
            
            if model_name not in model_raw_scores:
                continue
            
            raw_scores = model_raw_scores[model_name]
            
            if isinstance(raw_scores, dict):
                # Video model with multiple metrics
                # Step 1: Compute weighted sum of metrics for this model
                
                if per_model_metric_weights:
                    # Use per-model weights
                    metric_weights_to_use = per_model_metric_weights
                    # Weighted sum of metrics
                    model_metric_scores = []
                    for metric_name, weight in metric_weights_to_use.items():
                        metric_key = _normalize_key(metric_name)
                        if metric_key in raw_scores or metric_name in raw_scores:
                            # Try both normalized and original key
                            metric_values = raw_scores.get(metric_key, raw_scores.get(metric_name))
                            if metric_values is not None:
                                model_metric_scores.append([weight * s for s in metric_values])
                    
                    if model_metric_scores:
                        # Sum all weighted metrics for this model
                        model_sum = [sum(col) for col in zip(*model_metric_scores)]
                        # Step 2: Apply model weight
                        weighted_model_scores = [model_weight * s for s in model_sum]
                        weighted_scores_list.append(weighted_model_scores)
                else:
                    # No metric weights, use simple average of all metrics
                    all_metric_values = list(raw_scores.values())
                    if all_metric_values:
                        model_avg = [sum(col) / len(all_metric_values) for col in zip(*all_metric_values)]
                        weighted_model_scores = [model_weight * s for s in model_avg]
                        weighted_scores_list.append(weighted_model_scores)
            else:
                # Image model with single score
                # Directly apply model weight
                weighted_model_scores = [model_weight * s for s in raw_scores]
                weighted_scores_list.append(weighted_model_scores)
        
        # Final average: sum all weighted model scores
        if weighted_scores_list:
            total_scores = [sum(col) for col in zip(*weighted_scores_list)]
            output_scores['avg'] = total_scores
        else:
            # Fallback: if no weights matched, use simple average
            all_values = [v for k, v in output_scores.items() if k != 'avg']
            if all_values:
                total_scores = [sum(col) / len(all_values) for col in zip(*all_values)]
                output_scores['avg'] = total_scores
            else:
                # No scores at all - determine batch size from input
                if is_image_input:
                    batch_size = video_paths_or_images.shape[0]
                else:
                    batch_size = len(video_paths_or_images)
                output_scores['avg'] = [0.0] * batch_size
        assert 'avg' in output_scores.keys(), f"avg not in output_scores: {output_scores.keys()}"
        return output_scores, all_meta
    
    # Store reward models for offloading
    _fn._reward_models = reward_models
    return _fn

def get_reward_fn(args, device, logger):
    """
    Factory function to create reward function from args.
    
    Args:
        args: Training arguments. Should have:
            - reward_config (dict, optional): External reward configuration.
              Each remote reward model should specify its own server_url:
              {
                  "models": {
                      "tencent_remote": {
                          "weight": 1.0,
                          "server_url": "http://...",
                          "sub_reward": {...}
                      },
                      "video_align": {
                          "weight": 1.0,
                          "server_url": "http://...",
                          "sub_reward": {...}
                      }
                  }
              }
            - reward_model (str, optional): Reward model name (fallback if no reward_config)
            - reward_checkpoint_mode (str, optional): Checkpoint mode for local models
    
    Note:
        - Each remote reward model (video_align, video_score2, tencent_remote) should
          specify its own server_url in reward_config["models"][model_name]["server_url"]
        - The global args.remote_reward_url is no longer supported to avoid conflicts
          when using multiple remote reward models
    
    Returns:
        reward_fn: Unified reward function
    """
    # External config has highest priority
    if hasattr(args, "reward_config") and isinstance(args.reward_config, dict) and args.reward_config:
        reward_config = args.reward_config.copy()
        # Add reward_checkpoint_mode if not in config
        if hasattr(args, "reward_checkpoint_mode"):
            reward_config.setdefault("reward_checkpoint_mode", args.reward_checkpoint_mode)
        logger.info(f"Using external reward_config: {reward_config}")
    else:
        # Fallback: Build reward_config from reward_model only (default sub_reward weights)
        logger.warning("reward_config not provided, falling back to default config with reward_model only")
        
        reward_model = getattr(args, "reward_model", "videoalign_local")
        if reward_model == "auto":
            reward_model = "videoalign_local"
        
        # Use default sub_reward weights (all 1.0)
        sub_reward = {"VQ": 1.0, "MQ": 1.0, "TA": 1.0}
        
        # Build standard format reward_config
        reward_config = {
            "models": {
                reward_model: {
                    "weight": 1.0,
                    "sub_reward": sub_reward
                }
            }
        }
        
        # Model-specific config
        if reward_model == "videoalign_local":
            reward_config["reward_checkpoint_mode"] = getattr(args, "reward_checkpoint_mode", "none")
        elif reward_model in ["video_align", "video_score2", "tencent_remote"]:
            # These models require server_url, but we can't provide it without reward_config
            logger.warning(
                f"reward_model='{reward_model}' requires server_url, but reward_config is not provided. "
                f"Please provide reward_config with server_url for this model, or the model will fail to initialize."
            )
        
        # Add common configs if available
        if hasattr(args, "reward_checkpoint_mode"):
            reward_config.setdefault("reward_checkpoint_mode", args.reward_checkpoint_mode)
    
    logger.info(f"Creating reward function with config: {reward_config}")
    
    return multi_video_score(device, reward_config)

def create_reward_fn_from_config(config: Dict[str, Any], device, logger=None):
    """
    Factory function to create reward function from reward_config
    
    Args:
        args: Training arguments. Should have:
            - reward_config (dict, optional): External reward configuration.
              Each remote reward model should specify its own server_url:
              {
                  "models": {
                      "tencent_remote": {
                          "weight": 1.0,
                          "server_url": "http://...",
                          "sub_reward": {...}
                      },
                      "video_align": {
                          "weight": 1.0,
                          "server_url": "http://...",
                          "sub_reward": {...}
                      }
                  }
              }
            - reward_model (str, optional): Reward model name (fallback if no reward_config)
            - reward_checkpoint_mode (str, optional): Checkpoint mode for local models
    
    Note:
        - Each remote reward model (video_align, video_score2, tencent_remote) should
          specify its own server_url in reward_config["models"][model_name]["server_url"]
        - The global args.remote_reward_url is no longer supported to avoid conflicts
          when using multiple remote reward models
    
    Returns:
        reward_fn: Unified reward function
    """
    if logger:
        logger.info(f"Creating reward function from config: {config}")
    
    return multi_video_score(device, config)


# ============================================================================
# Test Functions
# ============================================================================

def test_tencent_remote(
    video_url: str,
    prompt: str,
    server_url: str = None,
    dimensions: List[str] = None,
    return_type: str = 'dict',
):
    """
    Test function for tencent_remote reward model using Tencent AutoEval 1.5.
    
    Args:
        video_url: URL or path to the video file (local path or HTTP URL)
        prompt: Text prompt for the video
        server_url: Optional server URL. If None, uses default:
                    'http://qsave-v1-5-1119.polaris:80/inference/'
        dimensions: List of dimensions to evaluate, e.g., ['T2V_Overall', 'TA'].
                    If None, defaults to ['T2V_Overall', 'TA']
        return_type: Return type, either 'dict' (default) or 'sum'
        
    Returns:
        scores_dict: Dictionary of scores with metric names as keys and lists of scores as values
        meta_dict: Additional metadata (empty dict)
        
    Example:
        >>> scores, meta = test_tencent_remote(
        ...     video_url="/path/to/video.mp4",
        ...     prompt="A beautiful sunset over the ocean",
        ...     dimensions=['T2V_Overall', 'TA']
        ... )
        >>> print(scores)
        {'T2V_Overall': [3.2], 'TA': [3.1], 'All': [3.2]}
    """
    import math
    
    print("=" * 80)
    print("Testing Tencent AutoEval 1.5 Remote Reward Model")
    print("=" * 80)
    print(f"Video URL/Path: {video_url}")
    print(f"Prompt: {prompt}")
    print(f"Server URL: {server_url if server_url else 'Using default (http://qsave-v1-5-1119.polaris:80/inference/)'}")
    print(f"Dimensions: {dimensions if dimensions else 'Using default ([\'T2V_Overall\', \'TA\'])'}")
    print(f"Return Type: {return_type}")
    print()
    
    # Initialize tencent_remote_score function
    try:
        reward_fn = tencent_remote_score(
            server_url=server_url,
            dimensions=dimensions,
            return_type=return_type,
        )
    except ValueError as e:
        print(f"配置错误: {e}")
        print(f"  请检查服务器 URL 格式是否正确")
        return {}, {}
    except Exception as e:
        print(f"初始化奖励函数时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}
    
    # Call the reward function
    # Note: video_paths and prompts must be lists
    video_paths = [video_url]
    prompts = [prompt]
    
    print("Calling reward function...")
    try:
        scores_dict, meta_dict = reward_fn(video_paths, prompts)
        
        print()
        print("=" * 80)
        print("Results")
        print("=" * 80)
        print(f"Scores Dictionary: {scores_dict}")
        print(f"Metadata: {meta_dict}")
        print()
        
        # Print individual metrics if available
        if scores_dict:
            print("Individual Metrics:")
            print("-" * 80)
            for metric_name, scores in scores_dict.items():
                if scores and len(scores) > 0:
                    score_value = scores[0]
                    if isinstance(score_value, (int, float)) and not math.isnan(score_value):
                        print(f"  {metric_name:20s}: {score_value:.4f}")
                    else:
                        print(f"  {metric_name:20s}: NaN (evaluation failed)")
            print()
            
            # Summary
            print("Summary:")
            print("-" * 80)
            valid_scores = {
                k: v[0] for k, v in scores_dict.items()
                if v and len(v) > 0 and isinstance(v[0], (int, float)) and not math.isnan(v[0])
            }
            if valid_scores:
                avg_score = sum(valid_scores.values()) / len(valid_scores)
                print(f"  Average Score: {avg_score:.4f}")
                print(f"  Number of Metrics: {len(valid_scores)}")
            else:
                print("  No valid scores available")
        else:
            print("No scores returned from the reward function.")
        
        print("=" * 80)
        
        return scores_dict, meta_dict
    
    except Exception as e:
        print()
        print("=" * 80)
        print("Error during testing")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return {}, {}


def test_hpsv3_remote(
    image_or_video_path: str,
    prompt: str,
    server_url: str = None,
    input_type: str = "auto",
):
    """
    Test function for hpsv3_remote reward model.
    Supports both images and videos.
    
    Args:
        image_or_video_path: Path to image or video file (local path or HTTP URL)
        prompt: Text prompt describing the image/video
        server_url: Server URL for the inference service. Format: "http://host:port" or "http://host:port/"
                    If None, raises ValueError.
        input_type: Type of input, either "auto" (default, auto-detect), "image", or "video"
        
    Returns:
        scores: numpy array of scores
        meta_dict: Additional metadata (empty dict)
        
    Example:
        >>> scores, meta = test_hpsv3_remote(
        ...     image_or_video_path="/path/to/video.mp4",
        ...     prompt="A beautiful sunset over the ocean",
        ...     server_url="http://28.59.19.23:8080"
        ... )
        >>> print(scores)
        [10.84]
    """
    import math
    
    print("=" * 80)
    print("Testing HPSv3 Remote Reward Model")
    print("=" * 80)
    print(f"Image/Video Path: {image_or_video_path}")
    print(f"Prompt: {prompt}")
    print(f"Server URL: {server_url if server_url else 'Not provided (required)'}")
    print(f"Input Type: {input_type}")
    print()
    
    # Validate server_url
    if server_url is None:
        print("Error: server_url is required for hpsv3_remote_score")
        print("  Please provide it in the format 'http://host:port'")
        return np.array([]), {}
    
    # Initialize hpsv3_remote_score function
    try:
        reward_fn = hpsv3_remote_score(device="cuda", server_url=server_url)
    except ValueError as e:
        print(f"配置错误: {e}")
        print(f"  请检查服务器 URL 格式是否正确")
        return np.array([]), {}
    except Exception as e:
        print(f"初始化奖励函数时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]), {}
    
    # Prepare input based on input_type
    if input_type == "auto":
        # Auto-detect: check file extension
        import os
        ext = os.path.splitext(image_or_video_path)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']
        is_video = ext in video_extensions
    elif input_type == "video":
        is_video = True
    elif input_type == "image":
        is_video = False
    else:
        print(f"Warning: Unknown input_type '{input_type}', using 'auto'")
        import os
        ext = os.path.splitext(image_or_video_path)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']
        is_video = ext in video_extensions
    
    if is_video:
        print(f"Detected input type: Video")
        inputs = [image_or_video_path]
    else:
        print(f"Detected input type: Image")
        inputs = image_or_video_path  # Will be converted to PIL in the function
    
    prompts = [prompt]
    metadata = None  # Optional metadata
    
    print("Calling reward function...")
    try:
        scores, meta_dict = reward_fn(inputs, prompts, metadata)
        
        print()
        print("=" * 80)
        print("Results")
        print("=" * 80)
        print(f"Scores: {scores}")
        print(f"Metadata: {meta_dict}")
        print()
        
        # Print individual scores if available
        if scores is not None and len(scores) > 0:
            print("Individual Scores:")
            print("-" * 80)
            for i, score in enumerate(scores):
                if isinstance(score, (int, float)) and not math.isnan(score):
                    print(f"  Sample {i+1:3d}: {score:.4f}")
                else:
                    print(f"  Sample {i+1:3d}: NaN (evaluation failed)")
            print()
            
            # Summary
            print("Summary:")
            print("-" * 80)
            valid_scores = [s for s in scores if isinstance(s, (int, float)) and not math.isnan(s)]
            if valid_scores:
                avg_score = np.mean(valid_scores)
                min_score = np.min(valid_scores)
                max_score = np.max(valid_scores)
                print(f"  Average Score: {avg_score:.4f}")
                print(f"  Min Score:     {min_score:.4f}")
                print(f"  Max Score:     {max_score:.4f}")
                print(f"  Valid Scores: {len(valid_scores)}/{len(scores)}")
            else:
                print("  No valid scores available")
        else:
            print("No scores returned from the reward function.")
        
        print("=" * 80)
        
        return scores, meta_dict
    
    except Exception as e:
        print()
        print("=" * 80)
        print("Error during testing")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return np.array([]), {}


def test_dynamic_degree(
    video_path: str,
    use_raft: bool = True,
    model_path: str = None,
    normalize: bool = True,
    target_fps: float = 8.0,
    save_flow_video: bool = False,
    output_path: str = None,
):
    """
    Test function for dynamic_degree reward model.
    Measures motion intensity in videos using optical flow (RAFT) or frame differences.
    
    Args:
        video_path: Path to video file
        use_raft: If True, use RAFT optical flow model (more accurate but requires model).
                  If False, use simple frame difference (lightweight, no model needed).
        model_path: Optional path to RAFT model checkpoint (only used if use_raft=True)
        normalize: If True, normalize scores to [0, 1] range
        target_fps: Target fps for frame extraction
        save_flow_video: If True, save optical flow visualization video
        output_path: Output path for flow video. If None, saves to same directory as input with '_flow.mp4' suffix
        
    Returns:
        scores_dict: Dictionary with 'dynamic_degree' key
        meta_dict: Additional metadata (empty dict)
        
    Example:
        >>> scores, meta = test_dynamic_degree(
        ...     video_path="/path/to/video.mp4",
        ...     use_raft=True,
        ...     save_flow_video=True
        ... )
        >>> print(scores)
        {'dynamic_degree': [0.75]}
    """
    import math
    import cv2
    from easydict import EasyDict as edict
    
    print("=" * 80)
    print("Testing Dynamic Degree Reward Model")
    print("=" * 80)
    print(f"Video Path: {video_path}")
    print(f"Use RAFT: {use_raft}")
    print(f"Model Path: {model_path if model_path else 'Using default'}")
    print(f"Normalize: {normalize}")
    print(f"Target FPS: {target_fps}")
    print(f"Save Flow Video: {save_flow_video}")
    print()
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist: {video_path}")
        return {}, {}
    
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if use_raft and save_flow_video:
        # Compute with flow visualization
        print("Computing optical flow with visualization...")
        
        from hymm.models.reward_models.raft import RAFT, InputPadder, flow_to_image, create_color_wheel_legend
        
        # Load RAFT model
        if model_path is None:
            model_path = REWARD_MODEL_PATH.get("raft", "")
            if not model_path or not os.path.exists(model_path):
                cache_path = os.path.expanduser("~/.cache/vbench/raft_model/models/raft-things.pth")
                if os.path.exists(cache_path):
                    model_path = cache_path
                else:
                    print(f"Error: RAFT model not found")
                    return {}, {}
        
        args = edict({
            "model": model_path,
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False
        })
        
        model = RAFT(args)
        ckpt = torch.load(model_path, map_location="cpu")
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(new_ckpt)
        model.to(device)
        model.eval()
        
        # Read video
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0
        
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        interval = max(1, int(round(fps / target_fps)))
        
        # Prepare output video writer
        if output_path is None:
            base_name = os.path.splitext(video_path)[0]
            output_path = f"{base_name}_flow.mp4"
        
        # Use mp4v codec for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps / interval  # Output fps matches extracted frame rate
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (width * 2, height))  # Side by side: original + flow
        
        # Create color wheel legend
        legend_size = min(120, height // 6)
        color_wheel = create_color_wheel_legend(size=legend_size, convert_to_bgr=True)
        
        # Extract frames
        frames = []
        frame_idx = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            
            if frame_idx % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb.astype(np.uint8)).permute(2, 0, 1).float()
                frame_tensor = frame_tensor[None].to(device)
                frames.append((frame, frame_tensor))  # Keep both BGR and tensor
            
            frame_idx += 1
        
        video.release()
        
        if len(frames) < 2:
            print(f"Warning: Not enough frames in video")
            return {}, {}
        
        # Compute optical flow and save visualization
        flow_scores = []
        print(f"Processing {len(frames)-1} frame pairs...")
        
        with torch.no_grad():
            for i in range(len(frames) - 1):
                frame_bgr1, image1 = frames[i]
                frame_bgr2, image2 = frames[i + 1]
                
                # Pad images
                padder = InputPadder(image1.shape)
                image1_padded, image2_padded = padder.pad(image1, image2)
                
                # Compute optical flow
                _, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
                
                # Get flow score
                flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
                u = flo[:, :, 0]
                v = flo[:, :, 1]
                rad = np.sqrt(np.square(u) + np.square(v))
                h, w = rad.shape
                rad_flat = rad.flatten()
                cut_index = int(h * w * 0.05)
                max_rad = np.mean(np.abs(np.sort(-rad_flat))[:cut_index])
                flow_scores.append(max_rad)
                
                # Visualize flow
                flow_vis = flow_to_image(flo, convert_to_bgr=True)
                
                # Resize flow visualization to match original frame size
                flow_vis_resized = cv2.resize(flow_vis, (width, height))
                
                # Concatenate original frame and flow visualization side by side
                combined = np.concatenate([frame_bgr1, flow_vis_resized], axis=1)
                
                # Add flow score text on left side (original video)
                score_text = f"Flow Score: {max_rad:.2f}"
                cv2.putText(combined, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add color wheel legend on right side (flow visualization) - top right corner
                legend_x = width * 2 - legend_size - 10
                legend_y = 10
                combined[legend_y:legend_y+legend_size, legend_x:legend_x+legend_size] = color_wheel
                
                # Add legend text labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                text_y = legend_y + legend_size + 15
                
                # Add direction labels around the color wheel
                cv2.putText(combined, "Color = Direction", (legend_x, text_y), font, font_scale, (255, 255, 255), thickness)
                cv2.putText(combined, "Bright = Fast", (legend_x, text_y + 15), font, font_scale, (255, 255, 255), thickness)
                cv2.putText(combined, "Dark = Slow/Static", (legend_x, text_y + 30), font, font_scale, (255, 255, 255), thickness)
                
                # Add direction hints
                hint_x = legend_x - 80
                hint_y = legend_y + legend_size // 2
                cv2.putText(combined, "Red->Right", (hint_x, hint_y - 20), font, 0.35, (0, 0, 255), 1)
                cv2.putText(combined, "Yellow->Up", (hint_x, hint_y), font, 0.35, (0, 255, 255), 1)
                cv2.putText(combined, "Green->Left", (hint_x, hint_y + 20), font, 0.35, (0, 255, 0), 1)
                cv2.putText(combined, "Blue->Down", (hint_x, hint_y + 40), font, 0.35, (255, 0, 0), 1)
                
                out.write(combined)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(frames)-1} frames")
        
        out.release()
        print(f"\nFlow video saved to: {output_path}")
        
        # Compute final score
        mean_score = np.mean(flow_scores)
        if normalize:
            normalized_score = 1.0 / (1.0 + np.exp(-(mean_score - 10) / 5))
            final_score = normalized_score
        else:
            final_score = mean_score
        
        scores_dict = {'dynamic_degree': [final_score]}
        
    else:
        # Use standard reward function
        try:
            if use_raft:
                print("Initializing RAFT-based dynamic degree scorer...")
                reward_fn = dynamic_degree_score(
                    device=device,
                    model_path=model_path,
                    normalize=normalize,
                    target_fps=target_fps
                )
            else:
                print("Initializing simple frame-difference based dynamic degree scorer...")
                reward_fn = dynamic_degree_score_simple(
                    device=device,
                    target_fps=target_fps,
                    normalize=normalize
                )
        except Exception as e:
            print(f"Error initializing reward function: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}
        
        # Call the reward function
        video_paths = [video_path]
        prompts = [""]  # Not used for dynamic_degree
        
        print("Computing dynamic degree score...")
        scores_dict, _ = reward_fn(video_paths, prompts)
    
    # Print results
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Scores Dictionary: {scores_dict}")
    print()
    
    # Print scores
    if 'dynamic_degree' in scores_dict:
        scores = scores_dict['dynamic_degree']
        print("Dynamic Degree Scores:")
        print("-" * 80)
        for i, score in enumerate(scores):
            import math
            if isinstance(score, (int, float)) and not math.isnan(score):
                print(f"  Video {i+1}: {score:.4f}")
                if normalize:
                    # Interpret the score
                    if score < 0.3:
                        motion_level = "Low (static/slow)"
                    elif score < 0.6:
                        motion_level = "Medium"
                    else:
                        motion_level = "High (fast motion)"
                    print(f"           Motion Level: {motion_level}")
            else:
                print(f"  Video {i+1}: NaN (evaluation failed)")
    
    print("=" * 80)
    
    return scores_dict, {}


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    import json
    
    # Add project root to PYTHONPATH if not already set
    # This allows the script to be run directly from any directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  For tencent_remote:  python rewards.py tencent <video_url> <prompt> [server_url] [dimensions_json] [return_type]")
        print("  For hpsv3_remote:    python rewards.py hpsv3 <image_or_video_path> <prompt> <server_url> [input_type]")
        print("  For dynamic_degree:  python rewards.py dynamic <video_path> [use_raft] [model_path] [normalize] [target_fps] [save_flow] [output_path]")
        print()
        print("Tencent Remote Arguments:")
        print("  video_url      : Path to video file or HTTP URL")
        print("  prompt         : Text prompt describing the video")
        print("  server_url     : (Optional) Server URL for inference service")
        print("  dimensions_json: (Optional) JSON string of dimensions list, e.g., '[\"T2V_Overall\",\"TA\"]'")
        print("  return_type    : (Optional) 'dict' or 'sum', default is 'dict'")
        print()
        print("HPSv3 Remote Arguments:")
        print("  image_or_video_path: Path to image or video file (local path or HTTP URL)")
        print("  prompt             : Text prompt describing the image/video")
        print("  server_url          : Server URL for inference service (required)")
        print("  input_type          : (Optional) 'auto' (default), 'image', or 'video'")
        print()
        print("Dynamic Degree Arguments:")
        print("  video_path    : Path to video file")
        print("  use_raft      : (Optional) 'true' for RAFT optical flow, 'false' for simple frame diff (default: true)")
        print("  model_path    : (Optional) Path to RAFT model checkpoint")
        print("  normalize     : (Optional) 'true' or 'false' (default: true)")
        print("  target_fps    : (Optional) Target fps for frame extraction (default: 24.0)")
        print("  save_flow     : (Optional) 'true' to save optical flow visualization video (default: false)")
        print("  output_path   : (Optional) Output path for flow video (default: <video_path>_flow.mp4)")
        print()
        print("Examples:")
        print("  # Test tencent_remote")
        print("  python rewards.py tencent /path/to/video.mp4 'A beautiful sunset'")
        print("  python rewards.py tencent /path/to/video.mp4 'A beautiful sunset' http://server:80/inference/")
        print()
        print("  # Test hpsv3_remote")
        print("  python rewards.py hpsv3 /path/to/video.mp4 'A beautiful sunset' http://28.59.19.23:8080")
        print("  python rewards.py hpsv3 /path/to/image.jpg 'A beautiful sunset' http://28.59.19.23:8080 image")
        print()
        print("  # Test dynamic_degree (measures motion intensity)")
        print("  python rewards.py dynamic /path/to/video.mp4")
        print("  python rewards.py dynamic /path/to/video.mp4 true")
        print("  python rewards.py dynamic /path/to/video.mp4 false  # Use simple frame diff (no RAFT needed)")
        print("  python rewards.py dynamic /path/to/video.mp4 true None true 24.0 true  # Save flow video")
        print("  python rewards.py dynamic /path/to/video.mp4 true None true 24.0 true /path/to/output_flow.mp4")
        sys.exit(1)
    
    test_mode = sys.argv[1].lower()
    
    if test_mode == "tencent":
        # Tencent remote test
        if len(sys.argv) < 4:
            print("Error: tencent mode requires at least <video_url> and <prompt>")
            sys.exit(1)
        
        video_url = sys.argv[2]
        prompt = sys.argv[3]
        server_url = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != 'None' else None
        dimensions = None
        return_type = 'dict'
        
        # Parse dimensions if provided
        if len(sys.argv) > 5 and sys.argv[5] != 'None':
            try:
                dimensions = json.loads(sys.argv[5])
                if not isinstance(dimensions, list):
                    print(f"Warning: dimensions must be a list, got {type(dimensions)}. Using default.")
                    dimensions = None
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse dimensions JSON: {e}. Using default.")
                dimensions = None
        
        # Parse return_type if provided
        if len(sys.argv) > 6:
            return_type = sys.argv[6]
            if return_type not in ['dict', 'sum']:
                print(f"Warning: return_type must be 'dict' or 'sum', got '{return_type}'. Using 'dict'.")
                return_type = 'dict'
        
        test_tencent_remote(video_url, prompt, server_url, dimensions, return_type)
    
    elif test_mode == "hpsv3":
        # HPSv3 remote test
        if len(sys.argv) < 5:
            print("Error: hpsv3 mode requires <image_or_video_path>, <prompt>, and <server_url>")
            sys.exit(1)
        
        image_or_video_path = sys.argv[2]
        prompt = sys.argv[3]
        server_url = sys.argv[4]
        input_type = sys.argv[5] if len(sys.argv) > 5 else "auto"
        
        test_hpsv3_remote(image_or_video_path, prompt, server_url, input_type)
    
    elif test_mode == "dynamic":
        # Dynamic degree test
        if len(sys.argv) < 3:
            print("Error: dynamic mode requires at least <video_path>")
            sys.exit(1)
        
        video_path = sys.argv[2]
        use_raft = True
        model_path = None
        normalize = True
        target_fps = 24.0
        save_flow = False
        output_path = None
        
        # Parse use_raft
        if len(sys.argv) > 3 and sys.argv[3] != 'None':
            use_raft = sys.argv[3].lower() in ['true', '1', 'yes', 'raft']
        
        # Parse model_path
        if len(sys.argv) > 4 and sys.argv[4] != 'None':
            model_path = sys.argv[4]
        
        # Parse normalize
        if len(sys.argv) > 5 and sys.argv[5] != 'None':
            normalize = sys.argv[5].lower() in ['true', '1', 'yes']
        
        # Parse target_fps
        if len(sys.argv) > 6 and sys.argv[6] != 'None':
            try:
                target_fps = float(sys.argv[6])
            except ValueError:
                print(f"Warning: Invalid target_fps '{sys.argv[6]}', using default 24.0")
                target_fps = 24.0
        
        # Parse save_flow
        if len(sys.argv) > 7 and sys.argv[7] != 'None':
            save_flow = sys.argv[7].lower() in ['true', '1', 'yes']
        
        # Parse output_path
        if len(sys.argv) > 8 and sys.argv[8] != 'None':
            output_path = sys.argv[8]
        
        test_dynamic_degree(video_path, use_raft, model_path, normalize, target_fps, save_flow, output_path)
    
    else:
        print(f"Error: Unknown test mode '{test_mode}'. Use 'tencent', 'hpsv3', or 'dynamic'")
        sys.exit(1)
