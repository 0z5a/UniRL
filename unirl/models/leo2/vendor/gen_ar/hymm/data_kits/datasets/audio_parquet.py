import random
import os.path as osp
from pathlib import Path
import torch
from typing import Dict
import pandas as pd
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ParquetDataset(Dataset):
    def __init__(self, source, data_root):
        self.data = pd.read_parquet(source)
        self.eval_data_root = data_root
    
    def __getitem__(self, index) -> Dict:
        # make sure the eval dataset folder structure is the same as listed below
        # make sure val.parquet contasin path that stores base name point to audio and video
        # eval_data_root
        #   - clip
        #   - t5_feat
        #   - videos
        #   - audios
        #   - val.parquet
        path = self.data.iloc[index]["path"]
        caption = self.data.iloc[index]["structure_caption"]
        t5_feat = osp.join(self.eval_data_root, "t5_feat", path + ".pt")
        clip_feat = osp.join(self.eval_data_root, "clip", path + ".pt")
        t5_feat = torch.load(t5_feat, "cpu")
        clip_feat = torch.load(clip_feat, "cpu")
        return {"prompt": caption, "t5_feat": t5_feat, "clip_feat": clip_feat, "path": path}

    def __len__(self):
        return len(self.data)
