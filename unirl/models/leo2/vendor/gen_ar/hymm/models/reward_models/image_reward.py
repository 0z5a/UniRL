# Image-Reward: Copyied from https://github.com/THUDM/ImageReward
import os
from typing import Union, List
from PIL import Image

import torch
try:
    import ImageReward as RM
except:
    pass
    # raise Warning("ImageReward is required to be installed (`pip install image-reward`) when using ImageReward for post-training.")


class ImageRewardModel(object):
    def __init__(self, model_name, device, http_proxy=None, https_proxy=None, med_config=None):
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        self.model_name = model_name if model_name else "ImageReward-v1.0"
        self.device = device
        self.med_config = med_config
        self.build_reward_model()

    def build_reward_model(self):
        self.model = RM.load(self.model_name, device=self.device, med_config=self.med_config)

    @torch.no_grad()
    def __call__(
            self,
            images,
            texts,
    ):
        if isinstance(texts, str):
            texts = [texts] * len(images)
        
        rewards = []
        for image, text in zip(images, texts):
            ranking, reward = self.model.inference_rank(text, [image])
            rewards.append(reward)
        return rewards


if __name__ == "__main__":
    prompt = "a painting of an ocean with clouds and birds, day time, low depth field effect"
    img_prefix = "/apdcephfs_nj10/share_301739632/yutaocui/workspace/data_processing/rl_data_proc/assets/images"
    generations = [f"{pic_id}.webp" for pic_id in range(1, 5)]
    img_list = [os.path.join(img_prefix, img) for img in generations] * 4
    
    # /apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/reward_model/image_reward/ImageReward.pt
    # /apdcephfs_zwfy/share_303793872/yutaocui/pretrained_models/image_reward/ImageReward.pt
    med_config = "/apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/reward_model/image_reward/med_config.json"
    image_reward = ImageRewardModel(
        model_name="/apdcephfs_zwfy/share_303793872/yutaocui/pretrained_models/image_reward/ImageReward.pt",
        device="cuda:1",
        med_config=med_config,
        http_proxy="http://star-proxy.oa.com:3128",
        https_proxy="http://star-proxy.oa.com:3128"
    )
    import time
    t1 = time.time()
    rewards = image_reward(img_list, prompt)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
    print(rewards)

"""
pip install image-reward
"""
# PYTHONPATH=./ python3 hymm/models/reward_models/image_reward.py