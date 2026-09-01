import io
import ast
from pathlib import Path
import random
import pyarrow as pa
import json

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from index_kits import ArrowIndexV2
from index_kits.resolution import ResolutionGroup
from processors.image_kits import read_local_image


class ArrowDataset(Dataset):
    def __init__(
            self,
            arrow_file,
            save_template,
            length=None,
            image_size=None,  # will resize image read from bytes
            column_dict=None,
            subset="",
            seed_type="auto",
            seed=1234,
            seed_plus=0,
            skip_exist=False,
            logger=None,
            image_processor=None,
    ):
        self.length = length
        self.seed_type = seed_type
        self.seed = seed
        self.seed_plus = seed_plus

        if logger is None:
            from loguru import logger

        self.table = pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file, "r")).read_all()
        if length is None:
            self.length = len(self.table)

        seed_col = column_dict.get("seed", "seed")
        if "seed" in column_dict:
            column_dict.pop("seed")
        logger.info(f"Using seed type: {self.seed_type}")
        if self.seed_type == "auto":
            if seed_col in self.table.column_names:
                self.seed_type = "file"
            else:
                self.seed_type = "fixed"
        if self.seed_type == "fixed":
            logger.info(f"    Using fixed seed for all samples: {seed}")
        elif self.seed_type == "random":
            logger.info(f"    Using random seed for all samples.")
        elif self.seed_type == "file":
            if seed_col not in self.table.column_names:
                raise ValueError(f"Missing seed column `{seed_col}` in table.")
            self.seed_col = seed_col
            logger.info(f"    Using seeds from column: {seed_col}")
        else:
            raise ValueError(f"Wrong seed_type: {self.seed_type}")
        
        # Select a subset of indices if needed.
        if len(subset) == 0:
            subset = list(range(len(self.table)))
        else:
            subset = list(map(int, subset.split()))
            logger.info(f"Generate a subset with ids: {subset}")

        # Build input dict
        self.total_dicts = []
        if isinstance(image_size, int):
            # During training, we always use image_size // 16 as step of resolution buckets; e.g. 1024 // 16 = 64
            resolutions = ResolutionGroup(image_size, image_size // 16)
        logger.info(f"Start loading data from {arrow_file}")
        for id_ in range(len(self.table)):
            if id_ > self.length:
                break
            if id_ not in subset:
                continue
            
            p_tmp = {
                "id": id_,
                "seed": self.get_seed(id_),
                "save_path": save_template.format(id_),
            }

            if skip_exist and Path(p_tmp["save_path"]).exists():
                continue
            
            for key, colunm_name in column_dict.items():
                if '@' not in colunm_name:
                    p_tmp[key] = self.table[colunm_name][id_].as_py()
                else:
                    real_colunm_name, *column_type = colunm_name.split('@')
                    if "ast" in column_type:
                        p_tmp[key] = torch.from_numpy(np.array(ast.literal_eval(self.table[real_colunm_name][id_].as_py())))
                    elif "bytes" in column_type:
                        image_bytes = io.BytesIO(self.table[real_colunm_name][id_].as_py())
                        image_bytes.seek(0)
                        pil_image = read_local_image(image_bytes, apply_exif=True)
                        if image_size is not None:
                            if isinstance(image_size, int):
                                # original width and height
                                original_width = pil_image.size[0]
                                original_height = pil_image.size[1]
                                # get the target size
                                tw, th = resolutions.get_target_size(original_width, original_height)
                                if id_ % 10 == 0:
                                    logger.info(f"Index {id_}: Resizing image from {original_width}x{original_height} to {tw}x{th} according to target size {image_size}")
                            # if image_size is a tuple 
                            elif len(image_size) == 2:
                                tw, th = image_size
                            # Use resize_and_crop, to align with the crop in InstructionTuningTransfusionArrowStream and indexer.py during training
                            resized_image, _ = ArrowIndexV2.resize_and_crop(
                                pil_image, 
                                target_size=(tw, th),
                                crop_type='center'
                            )
                            image_tensor = transforms.ToTensor()(resized_image)
                        elif image_processor is not None:
                            image_tensor = image_processor(pil_image)
                        else:
                            image_tensor = transforms.ToTensor()(pil_image)
                        if "mask" in key:
                            if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3:
                                image_tensor = image_tensor[0]
                        p_tmp[key] = image_tensor
                    elif "json" in column_type:
                        json_key = column_type[1]
                        caption = json.loads(self.table[real_colunm_name][id_].as_py())
                        p_tmp[key] = caption[json_key]
                    elif "random_choice" in column_type:
                        l = self.table[real_colunm_name][id_].as_py()
                        p_tmp[key] = random.choice(l)
                    elif "default_const" in column_type:
                        if real_colunm_name in self.table.column_names:
                            p_tmp[key] = self.table[real_colunm_name][id_].as_py()
                        else:
                            p_tmp[key] = column_type[1]
                    else:
                        raise ValueError(f"Unknown column type: {column_type}")

            self.total_dicts.append(p_tmp)

    def get_seed(self, idx, min_val=0, max_val=1_000_000):
        # Determine seeds
        if self.seed_type == "file":
            seed = int(self.table[self.seed_col][idx].as_py())
        elif self.seed_type == "fixed":
            seed = int(self.seed)
        elif self.seed_type == "random":
            seed = random.randint(min_val, max_val)
        else:
            raise ValueError(f"Wrong seed_type: {self.seed_type}")
        return seed + self.seed_plus

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.total_dicts[index]
