from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.models.multi_modal.mplug_for_all_tasks import MPlugForAllTasks
except ImportError as e:
    print(e)
    pipeline = None
    Tasks = None

from ..base_metric import BaseMetric


class DPGBenchMetric(BaseMetric):
    def __init__(self, vqa_model_path, dataset_name="DPGBench", max_size=1024, device=None):
        super().__init__()
        self.vqa_model_path = Path(vqa_model_path)
        self.rank = torch.distributed.get_rank()
        # We must explicitly set the device to the rank of the process, because modelscope doesn't use pytorch's
        # global rank.
        if device is None:
            if torch.distributed.is_initialized():
                device = torch.distributed.get_rank() % torch.cuda.device_count()
            else:
                device = 0
            self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if not self.vqa_model_path.exists():
            raise FileNotFoundError(f"{self.vqa_model_path} not found.")
        self._model = None
        self.transform = T.Compose([T.ToPILImage()])
        self.dataset_name = dataset_name
        self.max_size = max_size
        self.results = []

    def load_model(self, logger=None):
        if pipeline is None or Tasks is None:
            raise ImportError("modelscope is not installed. Please install it using `pip install modelscope`.")
        self._model = pipeline(Tasks.visual_question_answering, model=str(self.vqa_model_path), device=self.device)
        if logger is not None:
            logger.info("DPGBenchMetric: MPlug model loaded.")

    def release_model(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @torch.no_grad()
    def process(self, images, questions, dependencies, **kwargs):
        """
        Args:
            images (torch.Tensor or list of Image.Image): batch of image tensors with shape (B, 3, H, W)
                or list of PIL.Image with shape (H, W, 3)
            questions (list of dict): each dict contains question id and question text
            dependencies (list of dict): dict of dependency questions with question id as key
            **kwargs:
        """
        if isinstance(images, torch.Tensor):
            images = [self.transform(img) for img in images]
        assert all(isinstance(img, Image.Image) for img in images), "images should be Tensor or list of PIL Image"

        scores = []
        for image, question_dict, dependency_dict in zip(images, questions, dependencies):
            qid2scores = {}
            for qid, question in question_dict.items():
                answer = self.model({"image": image, "question": question})["text"]
                qid2scores[qid] = float(answer == "yes")
            for qid, parent_ids in dependency_dict.items():
                # zero-out scores if parent questions are answered 'no'
                any_parent_answered_no = False
                for parent_id in parent_ids:
                    if parent_id == 0:
                        continue
                    if qid2scores[str(parent_id)] == 0:
                        any_parent_answered_no = True
                        break
                if any_parent_answered_no:
                    qid2scores[qid] = 0

            score = sum(qid2scores.values()) / len(qid2scores)
            scores.append(score)
        self.results.append(np.array(scores))

    def compute_metrics(self, results):
        predictions = np.concatenate(results, axis=0)
        count = predictions.shape[0]
        print("DPGBenchMetric: predictions.shape is", count)

        return float(predictions.mean()), int(count)
