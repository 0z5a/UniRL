import json
import random
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


def prepare_dpg_data(raw_csv, save_data_path, save_question_path):
    # 'item_id', 'text', 'keywords', 'proposition_id', 'dependency', 'category_broad', 'category_detailed', 'tuple', 'question_natural_language'
    raw_data = pd.read_csv(raw_csv, header=0)

    random.seed(1234)

    # --------------------------------------------------
    # Prepare data csv for image sampling
    data = []
    for index, group in raw_data.groupby("item_id"):
        item = {
            "index": index,
            "seed": random.randint(0, 1000000),
            "prompt": group.text.iloc[0],
        }
        data.append(item)
    df = pd.DataFrame(data).sort_values(by="index")
    save_data_path = Path(save_data_path)
    save_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_data_path, index=False)
    print(f"Save data to {save_data_path}")

    # --------------------------------------------------
    # Prepare questions
    previous_id = ""
    question_dict = dict()

    for i, line in raw_data.iterrows():
        current_id = line.item_id
        qid = int(line.proposition_id)
        dependency_list_str = line.dependency.split(",")
        dependency_list_int = []
        for d in dependency_list_str:
            d_int = int(d.strip())
            dependency_list_int.append(d_int)

        if current_id == previous_id:
            question_dict[current_id]["qid2tuple"][qid] = line.tuple
            question_dict[current_id]["qid2dependency"][qid] = dependency_list_int
            question_dict[current_id]["qid2question"][qid] = line.question_natural_language
        else:
            question_dict[current_id] = dict(
                qid2tuple={qid: line.tuple},
                qid2dependency={qid: dependency_list_int},
                qid2question={qid: line.question_natural_language},
            )

        previous_id = current_id

    save_question_path = Path(save_question_path)
    save_question_path.parent.mkdir(parents=True, exist_ok=True)
    with save_question_path.open("w") as f:
        json.dump(question_dict, f, ensure_ascii=False, indent=4)
    print(f"Save question to {save_question_path}")


class DPGBenchDataset(Dataset):
    def __init__(self, csvs, debug=False):
        super().__init__()
        assert (
            isinstance(csvs, (tuple, list)) and len(csvs) == 2
        ), "csvs should be a list of two files: data.csv and questions.json"
        self.data = pd.read_csv(csvs[0], header=0)
        with open(csvs[1]) as f:
            self.vqa_data = json.load(f)
        if debug:
            # Randomly sample 96 samples for debugging
            self.data = self.data.sample(n=96, random_state=0).sort_values("index")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        questions = self.vqa_data[item["index"]]["qid2question"]
        dependencies = self.vqa_data[item["index"]]["qid2dependency"]
        ret = {
            "id": item["index"],
            "type": "prompt",
            "input": item["prompt"],
            "seed": item["seed"],
            "questions": questions,
            "dependencies": dependencies,
        }
        return ret

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        types = []
        inputs = []
        seeds = []
        questions = []
        dependencies = []

        for i in range(batch_size):
            ids.append(batch[i]["id"])
            types.append(batch[i]["type"])
            inputs.append(batch[i]["input"])
            seeds.append(batch[i]["seed"])
            questions.append(batch[i]["questions"])
            dependencies.append(batch[i]["dependencies"])

        ret = {
            "ids": ids,
            "type": types,
            "input": inputs,
            "seeds": seeds,
            "questions": questions,
            "dependencies": dependencies,
        }

        return ret


if __name__ == "__main__":
    prepare_dpg_data(
        raw_csv="__data/dpg_bench/dpg_bench.csv",
        save_data_path="data/dpg_bench/data.csv",
        save_question_path="data/dpg_bench/questions.json",
    )
