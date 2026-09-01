from pathlib import Path
from typing import Callable, Dict, Optional, Union
import pandas as pd
import numpy as np
import json
from copy import deepcopy
from loguru import logger
from processors.image_kits import read_local_image
import torch
from torch.utils.data import Dataset
import os

from hymm.constants import ASSETS_BASE


class VideoPromptDataset(Dataset):
    def __init__(self, args, logger,
        video_csv=None,
        text_encoder=None,
        text_encoder_2=None,
        seed=42,
        prompt_fn: Optional[Callable[[str, dict], dict]] = None,
        prompt_column: Union[str, Dict[str, float], None] = None,
    ):
        super().__init__()
        assert video_csv is not None, "video_csv is required"
        self.video_csv = video_csv
        self.args = args
        self.text_encoder=text_encoder
        self.text_encoder_2=text_encoder_2
        self.logger = logger
        self.seed = seed
        self.prompt_column_weights = self._normalize_prompt_column(prompt_column)
        self._repeat_random_map = None
        # self.name_mapper = lambda x: self.task_kwargs[x] if x in self.task_kwargs else x
        self.name_mapper = lambda x: x
        if prompt_fn is None:
            def default_prompt_fn(prompt, row):
                if args.prompt_prepend_fps:
                    fps = row['fps'] if 'fps' in row else args.video_fps
                    prompt = f"FPS:{fps}, " + prompt
                if args.prompt_prepend_content:
                    prompt = args.prompt_prepend_content + prompt
                if args.prompt_append_content:
                    prompt = prompt + args.prompt_append_content
                return {"role": "user", "content": prompt}
            prompt_fn = default_prompt_fn
        self.prompt_fn = prompt_fn
        self.load_data()

    def load_data(self):
        self.data = []
        for csv_path in self.video_csv:
            # Derive csv_file_name from the csv path (folder_name, file_stem)
            csv_file_name = os.path.splitext(os.path.basename(csv_path))[0]
            csv_folder_name = os.path.basename(os.path.dirname(csv_path))

            df = pd.read_csv(csv_path)
            self._sample_prompt_column(df, csv_path)
            self._build_message_list(df)

            # Handle index column
            if 'index' not in df.columns:
                df['index'] = range(len(df))
            
            # Handle seed column
            if 'seed' not in df.columns:
                df['seed'] = self.seed
            
            # Add csv_file_name column to track source CSV
            df['csv_file_name'] = [(csv_folder_name, f"{csv_file_name}") for i in range(len(df))]

            # Ensure stable column order
            required_cols = ['index', 'prompt', 'seed', 'message_list', 'csv_file_name']
            if 'ref_image_path' in df.columns:
                required_cols.append('ref_image_path')
            df = df[required_cols]
            self.data.extend(df.to_dict('records'))
        self.total_length = len(self.data)
                    
    def __len__(self):
        if hasattr(self, '_repeat_random_map') and self._repeat_random_map is not None:
            return len(self._repeat_random_map)
        return len(self.data)
         
    def get_text_tokens(self, text_encoder, description, text_len):
        text_inputs = text_encoder.text2tokens(description, data_type='video', max_length=text_len)
        text_ids = text_inputs["input_ids"].squeeze(0)
        text_mask = text_inputs["attention_mask"].squeeze(0)
        return text_ids, text_mask

    def __getitem__(self, idx):
        if hasattr(self, '_repeat_random_map') and self._repeat_random_map is not None:
            return self._repeat_random_map[idx]
        # key: index, prompt, seed
        sample = self.data[idx]
        
        result = (
            sample['index'],
            sample['prompt'],
            int(sample['seed']) if isinstance(sample['seed'], str) else sample['seed'],
            sample.get('ref_image_path', ""),
            sample.get('message_list'),
            sample.get('csv_file_name'),
        )
        return result 
    
    @staticmethod
    def _normalize_prompt_column(prompt_column):
        if prompt_column is None:
            return {"prompt": 1.0}
        if isinstance(prompt_column, str):
            return {prompt_column: 1.0}
        if isinstance(prompt_column, dict):
            assert len(prompt_column) > 0, "prompt_column dict must not be empty"
            for k, v in prompt_column.items():
                assert isinstance(k, str), f"prompt_column key must be str, got {type(k)}"
                assert isinstance(v, (int, float)) and v >= 0, \
                    f"prompt_column value must be a non-negative number, got {v} for key {k}"
            total = float(sum(prompt_column.values()))
            assert total > 0, "prompt_column probabilities must sum to a positive value"
            assert abs(total - 1.0) < 1e-6, \
                f"prompt_column probabilities must sum to 1, got {total}"
            return {k: float(v) / total for k, v in prompt_column.items()}
        raise TypeError(
            f"prompt_column must be None, str, or dict[str, float], got {type(prompt_column)}"
        )

    def _sample_prompt_column(self, df, csv_path):
        cols = list(self.prompt_column_weights.keys())
        probs = list(self.prompt_column_weights.values())
        missing = [c for c in cols if c not in df.columns]
        assert missing == [], (
            f"Missing prompt columns {missing} in csv {csv_path}, "
            f"available columns: {list(df.columns)}"
        )

        if len(cols) == 1:
            picked_col = cols[0]
            if picked_col != 'prompt':
                if 'prompt' in df.columns:
                    df.drop(columns=['prompt'], inplace=True)
                df.rename(columns={picked_col: 'prompt'}, inplace=True)
            return

        rng = np.random.default_rng(self.seed)
        picks = rng.choice(cols, size=len(df), p=probs)
        sampled = [df.iloc[i][col] for i, col in enumerate(picks)]
        df['prompt'] = sampled

    def _build_message_list(self, df):
        if (message_col := self.name_mapper("message_list")) in df.columns:
            # OpenAI format message list
            df[message_col] = df[message_col].apply(json.loads)
        
        elif (prompt_col := self.name_mapper("prompt")) in df.columns and \
            ((src_col := self.name_mapper("src_img_path")) in df.columns or (count_col := self.name_mapper("count")) in df.columns):

            src_col = self.name_mapper("src_img_path")
            count_col = self.name_mapper("count")
            if count_col not in df.columns:
                df["count"] = [1] * len(df)
                count_col = "count"

            df["message_list"] = df.apply(
                lambda row: 
                    [{
                        "role": "user",
                        "content": [{
                            "type": "image",
                            "image": self.format_file_path(
                                row[src_col if src_col in df.columns else self.name_mapper("src_img_path_1")]
                            )
                        }]
                    }] +
                    [
                        {
                            "role": "user",
                            "content": [{
                                "type": "image",
                                "image": self.format_file_path(row[self.name_mapper(f"src_img_path_{i+1}")])
                            }]
                        } for i in range(1, row[count_col])
                    ] + 
                    [
                        self.prompt_fn(row[prompt_col], row),
                    ],
                axis=1,
            )

        elif (prompt_col := self.name_mapper("prompt")) in df.columns:
            df["message_list"] = df.apply(lambda row: [self.prompt_fn(row[prompt_col], row)], axis=1)

        else:
            raise NotImplementedError(
                f"[MessageListDataset] Unsupported CSV dataset format with columns: {df.columns}."
            )

    def prepare_repeat_random_slots(self, num_generations, world_size, global_seed=42):
        # get all data
        prompt_list = [x['prompt'] for x in self.data]
        index_list = [x['index'] for x in self.data]
        seed_list = [int(x['seed']) for x in self.data]
        total = len(prompt_list)
        slot_list = []
        # slot: prompt_id * num_generations + sample_id
        for pidx in range(total):
            for s in range(num_generations):
                slot_list.append({
                    "prompt_index": pidx,
                    "index": index_list[pidx],
                    "prompt": prompt_list[pidx],
                    "seed": seed_list[pidx] + s * 10000,   # strong consistency offset, avoid different slot seed conflict
                    "sample_slot": s,
                })
        # pad to world_size
        if len(slot_list) < world_size:
            if not slot_list:
                slot_list = [None] * world_size  # 或者根据业务需求设置其他默认值
            else:
                slot_list += [slot_list[-1]] * (world_size - len(slot_list))
        self._repeat_random_map = slot_list

    @staticmethod
    def format_file_path(file_path: Optional[str]):
        if file_path is None:
            return None
        assert isinstance(file_path, str), f"file_path must be str, but got {type(file_path)}"

        file_path = file_path.strip()
        if file_path == "" or file_path.startswith("/") or file_path.startswith("http"):
            return file_path

        # If relative path, Prepend the ASSETS_BASE path
        file_path = Path(ASSETS_BASE) / file_path
        assert file_path.exists(), f"{file_path} does not exist"
        return str(file_path)

    def get_sample_by_index(self, index):
        if index < 0 or index >= len(self):
            return None

        if hasattr(self, '_repeat_random_map') and self._repeat_random_map is not None:
            return self._repeat_random_map[index]

        return self.data[index] if index < len(self.data) else None

