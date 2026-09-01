import json
import random
from pathlib import Path

import torch
from torchvision.transforms import transforms
from PIL import Image
from index_kits import ArrowIndexV2

from hymm.utils.helpers import to_2tuple


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, test, dataset_name, image_base, target_size, max_new_tokens,
                 pad_color=(127, 127, 127), get_item_callback=None):
        self.test = Path(test).read_text().splitlines()
        self.dataset_name = dataset_name
        self.image_base = Path(image_base)
        self.target_size = to_2tuple(target_size)
        self.pad_color = pad_color

        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )
        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": max_new_tokens}
        if get_item_callback is None:
            self.get_item_callback = lambda x: x
            self.process_image = True
        else:
            self.get_item_callback = get_item_callback
            self.process_image = False

    def __len__(self):
        return len(self.test)

    def get_coco_image_path(self, image_path):
        image_file = Path(image_path).name
        return self.image_base / image_file

    def __getitem__(self, index):
        is_dummy = index // len(self) > 0
        index = index % len(self)

        data = json.loads(self.test[index].strip())
        image_path, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)
        if self.dataset_name == "vqav2_val":
            image_path = self.get_coco_image_path(image_path)

        image = Image.open(image_path).convert("RGB")
        if self.process_image:
            image, _ = ArrowIndexV2.resize_and_pad(
                image, self.target_size, resample=Image.Resampling.BICUBIC, pad_color=self.pad_color,
            )
            image = self.pil_image_to_tensor(image)

        ret = self.get_item_callback({
            'ids': question_id,
            'image': image,
            'prompt': question,
            'seeds': random.randint(0, 1_000_000),
            "is_dummy": is_dummy,
        })

        return ret
