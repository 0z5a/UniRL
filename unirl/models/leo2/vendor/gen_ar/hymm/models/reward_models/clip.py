from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from torch import nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor, InterpolationMode


def get_clip(args: Optional[dict] = None) -> nn.Module:
    available_path = dict(zw="/apdcephfs_zwfy/share_303937731/yutaocui/models_zoo/hps_ckpt/open_clip_pytorch_model.bin",
                          nj="/apdcephfs_nj10/share_301739632/yutaocui/workspace/DanceGRPO/hps_ckpt/open_clip_pytorch_model.bin",
                          gy="/apdcephfs_gy2/share_302507476/lucazzliu/misc/ckpts/hpsv2/open_clip_pytorch_model.bin",
                          bj="/apdcephfs_bjzf/share_303693591/lucazzliu/misc/ckpts/hpsv2/open_clip_pytorch_model.bin",
                          gz="/apdcephfs_fsgm/share_303793872/lucazzliu/misc/ckpts/hpsv2/open_clip_pytorch_model.bin")
    try:
        assert hasattr(args, "reward_model_region") and args.reward_model_region in available_path.keys(), \
            f"something went wrong with args.reward_model_region"
        reward_model_region = args.reward_model_region
    except AssertionError:
        reward_model_region = 'zw'
    reward_model = clip_reward(available_path[reward_model_region])
    return reward_model


class clip_reward(nn.Module):
    def __init__(self, clip_path):
        super().__init__()
        self.model, self.tokenizer, self.preprocess_val = CLIP(clip_path)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)


    def preprocess_tensor(self, image):
        # input 0-1 tensors
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        resize_size = 224  # 你这里resize和crop都是224
        # image encode
        def _transform():
            transform = Compose([
                Resize((resize_size, resize_size), interpolation=InterpolationMode.BICUBIC),
                Normalize(std=image_std,mean=image_mean),
            ])
            return transform


        rm_preprocess = _transform()
        image = rm_preprocess(image)
        return image.to(torch.float32)

    def get_fine_image_features(self, images, conduct_preprocess=True):
        self.model.visual.output_tokens = True
        if conduct_preprocess:

            images = self.preprocess_tensor(images)
        with torch.cuda.amp.autocast():
            cls, feat256 = self.model.visual(images)
            cls = F.normalize(cls, dim=-1)
        self.model.visual.output_tokens = False
        return cls, feat256

    def correlation(self, image_features, text_features):
        with torch.cuda.amp.autocast():
            logits_per_image = image_features @ text_features.T
            clip_score = torch.diagonal(logits_per_image)
        return clip_score

    def checkpointing(self, do_checkpoint=True):
        self.model.gradient_checkpointing = do_checkpoint

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def requires_grad_(self, requires_grad: bool):
        self.model.requires_grad_(requires_grad)

def CLIP(clip_path, device: str = "cuda"):
    model_dict = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        clip_path,
        precision='amp',
        # device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )

    processor = get_tokenizer('ViT-H-14')
    reward_model = model.to(device).eval()
    return reward_model, processor, preprocess_val
