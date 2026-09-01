from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import PIL.Image
import numpy as np
import pandas as pd
import torch
from ..base_metric import BaseMetric
from hymm.models.reward_models.objs_counting_groundingdino import ObjectsCountingGroundingDino


class CountingEvalMetric(BaseMetric):
    def __init__(self, dataset_name="CountingEval", box_thre=0.5, max_size=1024, device=None):
        super().__init__()
        if device is None:
            if torch.distributed.is_initialized():
                device = torch.distributed.get_rank() % 8
            else:
                device = 0
            self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # We must explicitly set the device to the rank of the process, because modelscope doesn't use pytorch's
        # global rank.
        self.dataset_name = dataset_name
        self.results = []
        self.max_size = max_size
        self.box_thre = box_thre
        self.obj_count_model = None

    def load_model(self, logger=None):
        self.obj_count_model = ObjectsCountingGroundingDino(box_thre=self.box_thre, device=self.device)
        logger.info("GroundingDino model for counting_eval metric loaded.")

    def release_model(self):
        self.obj_count_model = None

    @staticmethod
    def tensors2pil_images(images):
        """
        images can be:
        * torch.Tensor, shape (B, C, H, W)
        * numpy.ndarray, values in interval [0, 1], shape (H, W, C) or (B, H, W, C)
        * a list of PIL.Image.
        """
        if isinstance(images, torch.Tensor):
            # Tensor -> numpy.ndarray
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        if isinstance(images, np.ndarray):
            # numpy.ndarray -> PIL.Image
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            if images.shape[-1] == 1:
                # special case for grayscale (single channel) images
                images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in images]
            else:
                images = [PIL.Image.fromarray(image) for image in images]

        return images

    @torch.no_grad()
    def process(self, images, metadata, **kwargs):
        """
        Args:
            images (torch.cuda.Tensor): batch of image tensors with shape (B, 3, H, W) and interval [0, 1]
                or list of image paths.
            metadata (list of dict): each dict contains the metadata for the corresponding image
            **kwargs:
        """

        if isinstance(images[0], torch.Tensor):
            pil_images = self.tensors2pil_images(images)
        objs = [md_i["obj"] for md_i in metadata]
        gt_nums = [md_i["count"] for md_i in metadata]
        pred_nums = self.obj_count_model.pred_objs_num(pil_images, objs=objs)
        scores = []
        for gt_num, pred_num in zip(gt_nums, pred_nums):
            is_correct = gt_num == pred_num
            score = {
                'correct': is_correct,
                'tag': gt_num,
            }
            scores.append(score)

        self.results.extend(scores)

    def compute_metrics(self, results):
        df = pd.DataFrame(results)
        # df.to_csv('my_counting_eval_results.csv')
        out_dict = {}

        for tag, group in sorted(df.groupby('tag')):
            tag_score = group['correct'].mean()
            out_dict[tag] = (round(float(tag_score), 6), len(group))

        count = len(df)
        avg_accuracy = df['correct'].mean()  # Calculate accuracy directly from df
        out_dict['avg'] = (
            round(float(avg_accuracy), 6),
            count
        )
        return out_dict
