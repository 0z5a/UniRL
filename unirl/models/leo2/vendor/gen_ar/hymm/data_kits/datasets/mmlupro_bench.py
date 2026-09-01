import os
import random

import torch
import pandas as pd
from torch.utils.data import Dataset


CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
INITIAL_PROMPT = """The following are multiple choice questions (with answers) about {}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice."""
DEFAULT_MAX_LENGTH = 4096
DEFAULT_NTRAIN = 5
INVALID_ANS = "[invalid]"
TEST_FILE = "test-00000-of-00001.parquet"
VALID_FILE = "validation-00000-of-00001.parquet"
random.seed(42)

class MMLUProBenchDataset(Dataset):
    """MMLU-Pro Dataset with dynamic prompt generation and length control.

    Args:
        data_dir (str): Path to the data directory.
        ntrain (int): Number of few-shot examples to use.
        max_length (int): Maximum sequence length (default: 4096).
    """

    def __init__(self, data_dir=None, ntrain=DEFAULT_NTRAIN, max_length=DEFAULT_MAX_LENGTH):
        super().__init__()
        self.data_dir = data_dir
        self.ntrain = ntrain
        self.max_length = max_length
        self.data = self._load_data(TEST_FILE)
        self.val_data = self._load_data(VALID_FILE)

        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": 1024, "total_count": len(self.data)}

    def _load_data(self, file_name):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")

        file = os.path.join(self.data_dir, file_name)    
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} does not exist.")

        # Load the data from the Parquet file
        data = pd.read_parquet(file)
        return self._preprocess(data.to_dict(orient="records"))

    def _preprocess(self, data):
        """Preprocess the dataset to filter and format options."""
        processed_data = []
        for example in data:
            options = [opt for opt in example["options"] if opt != "N/A"]
            example["options"] = options
            example["seed"] = random.randint(0, 1000000)
            processed_data.append(example)
        return processed_data

    def _format_example(self, example, include_answer=True):
        """Format a single example into a prompt."""
        prompt = "Question:\n"
        prompt += example["question"] + "\n"
        prompt += "Options:\n"
        for i, opt in enumerate(example["options"]):
            prompt += f"{CHOICES[i]}. {opt}\n"
        if include_answer:
            # print(example["cot_content"])
            cot_content = example["cot_content"].replace(
                "A: Let's think step by step.", "Answer: Let's think step by step."
            )
            prompt += cot_content + "\n\n"
        else:
            prompt += "Answer: Let's think step by step."
        return prompt

    def _generate_cot_prompt(self, val_data, current_example, k):
        """Generate a few-shot prompt."""
        prompt = ""
        subject = current_example["category"]
        val_data = [ex for ex in val_data if ex["category"] == subject][:k]
        prompt = INITIAL_PROMPT.format(subject) + "\n"
        for example in val_data:
            prompt += self._format_example(example, include_answer=True)
        prompt += self._format_example(current_example, include_answer=False)
        return prompt

    def __len__(self):
        return len(self.data)  # 12032

    def __getitem__(self, idx):
        """Return a single sample with prompt and metadata."""
        is_dummy = idx // len(self) > 0
        idx = idx % len(self)

        current_example = self.data[idx]
        val_data = [ex for ex in self.val_data if ex["category"] == current_example["category"]]
        k = min(self.ntrain, len(val_data))
        prompt = self._generate_cot_prompt(val_data, current_example, k)

        # Check length and truncate if necessary
        while len(prompt.split()) > self.max_length and k > 0:
            k -= 1
            prompt = self._generate_cot_prompt(val_data, current_example, k)

        return {
            "id": idx,
            "type": "prompt",
            "seed": current_example["seed"],
            "input": prompt,
            "subjects": current_example["category"],
            "labels": current_example["answer"],
            "options": current_example["options"],
            "is_dummy": is_dummy,
        }

    @staticmethod
    def collate_fn(batch):
        """Collate function to combine multiple samples into a batch."""
        ids = [item["id"] for item in batch]
        inputs = [item["input"] for item in batch]
        subjects = [item["subjects"] for item in batch]
        labels = [item["labels"] for item in batch]
        options = [item["options"] for item in batch]
        types = [item["type"] for item in batch]
        seeds = [item["seed"] for item in batch]
        is_dummy = torch.tensor([item["is_dummy"] for item in batch])

        return {
            "ids": ids,
            "type": types,
            "seeds": seeds,
            "input": inputs,
            "subjects": subjects,
            "labels": labels,
            "options": options,
            "is_dummy": is_dummy,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = MMLUProBenchDataset(data_dir="data/mmlu_pro", ntrain=5, max_length=4096)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=MMLUProBenchDataset.collate_fn)

    print(f"Number of samples in the dataset: {len(dataset)}")
    # Iterate through the DataLoader
    for batch in dataloader:
        print(batch)
        if batch["ids"][0] == 5:
            break

# PYTHONPATH="." python3 hymm/data_kits/mmlupro_bench.py
