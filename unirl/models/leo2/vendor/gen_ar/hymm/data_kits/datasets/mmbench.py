import os
import random
import string

import torch
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image
from index_kits import ArrowIndexV2

try:
    from vlmeval.dataset import ImageMCQDataset
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"{e}. Please install VLMEvalKit first.")

from hymm.utils.helpers import to_2tuple


def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False


class MMBenchDataset(torch.utils.data.Dataset):

    def __init__(self, LMUDataRoot, dataset_name, target_size, pad_color=(127, 127, 127)):
        self.dataset_name = dataset_name
        self.target_size = to_2tuple(target_size)
        self.pad_color = pad_color

        os.environ["LMUData"] = LMUDataRoot
        self.dataset = ImageMCQDataset(dataset_name)

        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )
        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": 512}

    def __len__(self):
        return len(self.dataset)

    def build_prompt(self, line):
        tgt_path = self.dataset.dump_image(line)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def __getitem__(self, idx):
        line = self.dataset.data.iloc[idx]
        question_id = int(line['index'])
        item = self.build_prompt(line)
        assert len(item) == 2, f"Item should contain two elements, got {len(item)}. {item}"
        image_path = item[0]['value']
        question = item[1]['value']

        image = Image.open(image_path).convert("RGB")
        image, _ = ArrowIndexV2.resize_and_pad(
            image, self.target_size, resample=Image.Resampling.BICUBIC, pad_color=self.pad_color,
        )
        image_tensor = self.pil_image_to_tensor(image)

        return {
            'id': question_id,
            'image': image_tensor,
            'prompt': question,
            'seed': random.randint(0, 1_000_000),
            'line': line,   # contain all the information of the sample
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        seeds = []
        prompts = []
        lines = []

        images = torch.stack([sample["image"] for sample in batch], 0)
        for i in range(batch_size):
            ids.append(batch[i]["id"])
            seeds.append(batch[i]["seed"])
            prompts.append(batch[i]["prompt"])
            lines.append(batch[i]["line"])

        ret = {
            "ids": ids,
            "image": images,
            "prompt": prompts,
            "seeds": seeds,
            "lines": pd.DataFrame(lines),
        }

        return ret
