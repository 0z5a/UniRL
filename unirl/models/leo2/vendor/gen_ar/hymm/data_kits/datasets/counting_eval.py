import json
import pandas as pd
from torch.utils.data import Dataset


class CountingEvalDataset(Dataset):
    def __init__(self, jsonl, debug=False):
        super().__init__()
        data = []
        with open(jsonl, "r") as f:
            for line in f:
                item = json.loads(line)
                metadata = {
                    "tag": item["tag"],
                    "prompt": item["prompt"], 
                    "obj": item["include"][0]["class"],
                    "count": item["include"][0]["count"],
                }
                if "exclude" in item:
                    metadata["exclude"] = item["exclude"]
                data.append({
                    "index": item['index'],
                    "prompt": item["prompt"],
                    "seed": item["seed"],
                    "metadata": metadata,
                })
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        ret = {
            "id": item["index"],
            "type": "prompt",
            "input": item["prompt"],
            "seed": item["seed"],
            "metadata": item["metadata"],
        }
        return ret

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        types = []
        inputs = []
        seeds = []
        metadata = []

        for i in range(batch_size):
            ids.append(batch[i]["id"])
            types.append(batch[i]["type"])
            inputs.append(batch[i]["input"])
            seeds.append(batch[i]["seed"])
            metadata.append(batch[i]["metadata"])

        ret = {
            "ids": ids,
            "type": types,
            "input": inputs,
            "seeds": seeds,
            "metadata": metadata,
        }

        return ret


if __name__ == "__main__":
    counteval_dataset = CountingEvalDataset(jsonl="data/counting_eval/counting_eval.jsonl")
    print(counteval_dataset.data)