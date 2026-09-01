from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class LabelDataset(Dataset):
    def __init__(self,
                 labels,
                 save_template,
                 seed=1234,
                 skip_exist=False,
                 logger=None,
                 **kwargs,
                 ):
        if isinstance(labels, int):
            labels = list(range(labels))
        assert isinstance(labels, list), f"Wrong labels type: {type(labels)}"
        seeds = [seed + i for i in range(len(labels))]

        data = []
        for label, sd in zip(labels, seeds):
            data.append({
                "label": label,
                "seed": sd,
            })

        self.table = pd.DataFrame(data)

        # Build input dict
        self.total_data = []
        self.total_data_dicts = []
        for id_, item in self.table.iterrows():
            self.total_data.append(str(item["label"]))

            p_tmp = {
                "id": item["label"],
                "type": "label",
                "input": item["label"],
                "seed": item["seed"],
                "save_path": save_template.format(item["label"]),
            }

            if skip_exist and Path(p_tmp["save_path"]).exists():
                continue
            self.total_data_dicts.append(p_tmp)

    def __len__(self):
        return len(self.total_data_dicts)

    def __getitem__(self, index):
        return self.total_data_dicts[index]

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        types = []
        inputs = []
        seeds = []

        for i in range(batch_size):
            ids.append(batch[i]["id"])
            types.append(batch[i]["type"])
            inputs.append(batch[i]["input"])
            seeds.append(batch[i]["seed"])

        ret = {
            "ids": ids,
            "type": types,
            "input": inputs,
            "seeds": seeds,
        }

        return ret
