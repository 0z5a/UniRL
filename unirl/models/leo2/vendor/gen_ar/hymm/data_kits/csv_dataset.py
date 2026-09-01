import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from loguru import logger

from .datasets import DATASETS, load_dataset
from hymm.constants import TESTSET_TEMPLATE, CAPTION_TEST_PATH, ASSETS_BASE
from hymm.utils.torch_utils import except_collate_fn
from processors.image_kits import read_local_image


def decode_csv_file(csv, error_message=''):
    if csv is None:
        raise ValueError(f"File not found: {csv}. {error_message}")
    if Path(csv).suffix == "":
        # When src_file has no suffix, we treat it as a stem and fill in the template.
        csv = Path(TESTSET_TEMPLATE.format(csv))
    else:
        csv = Path(csv)
    if not csv.exists():
        raise FileNotFoundError(f"File not found: {csv}. {error_message}")
    return csv


class CSVDataset(Dataset):
    def __init__(self,
                 source,
                 save_template,
                 subset="",
                 seed_type="auto",
                 seed=1234,
                 seed_plus=0,
                 pos_prompt="",
                 skip_exist=False,
                 logger=None,
                 extra_cols=None,
                 callback=None,
                 get_item_callback=None,
                 **kwargs,
                 ):
        self.seed_type = seed_type
        self.seed = seed
        self.seed_plus = seed_plus

        # Check csv source type
        attr_dict = {}
        source_parts = source.split("@@")

        # Fill source file
        src_file = decode_csv_file(source_parts[0])
        if len(source_parts) > 1:
            for part in source_parts[1:]:
                key, value = part.split("=")
                attr_dict[key] = value

        # Load dataset file
        suffix = src_file.suffix.lstrip(".")
        if suffix == "csv":
            self.table = pd.read_csv(src_file, keep_default_na=False, **kwargs)
        elif suffix == "jsonl":
            self.table = pd.read_json(src_file, lines=True, **kwargs)
        elif suffix == "xlsx":
            self.table = pd.read_excel(src_file, **kwargs)
        else:
            raise ValueError(f"Wrong dataset file suffix: {suffix}")

        # Check index_col exists (optional)
        index_col = attr_dict.get("index", "index")
        if index_col in self.table.columns:
            self.table.set_index(index_col, inplace=True, drop=False)

        # Check text_col exists (mandatory)
        text_col = attr_dict.get("text", "prompt")
        if text_col not in self.table.columns:
            raise ValueError(f"Missing text column `{text_col}` in table. {self.table.columns}")

        # Check seed_col exists (optional)
        seed_col = attr_dict.get("seed", "seed")
        logger.info(f"Using seed type: {self.seed_type}")
        if self.seed_type == "auto":
            if seed_col in self.table.columns:
                self.seed_type = "file"
            else:
                self.seed_type = "fixed"
        if self.seed_type == "fixed":
            logger.info(f"    Using fixed seed for all samples: {seed}")
        elif self.seed_type == "random":
            logger.info(f"    Using random seed for all samples.")
        elif self.seed_type == "file":
            if seed_col not in self.table.columns:
                raise ValueError(f"Missing seed column `{seed_col}` in table.")
            self.seed_col = seed_col
            logger.info(f"    Using seeds from column: {seed_col}")
        else:
            raise ValueError(f"Wrong seed_type: {self.seed_type}")

        # Extra columns
        if not extra_cols:
            extra_cols = []

        # Processor
        if not callback:
            callback = lambda x: x
        
        if not get_item_callback:
            get_item_callback = lambda x: x
        self.get_item_callback = get_item_callback

        # Select a subset of indices if needed.
        if len(subset) == 0:
            subset = list(self.table.index)
        else:
            subset = list(map(int, subset.split()))
            logger.info(f"Generate a subset with ids: {subset}")

        # Build input dict
        self.total_prompts = []
        self.total_prompt_dicts = []
        for id_, item in tqdm(self.table.iterrows(), total=len(self.table), desc="Building csv dataset"):
            if id_ not in subset:
                continue
            if not item[text_col]:
                logger.info(f"Empty prompt for id {id_}")
            else:
                prompt = str(item[text_col]) + pos_prompt
                self.total_prompts.append(prompt)

                p_tmp = callback({
                    "id": id_,
                    "type": "prompt",
                    "input": prompt,
                    "seed": self.get_seed(id_),
                    "save_path": save_template.format(id_),
                    **{
                        f"extra_{col}": item[col] if col in item else None for col in extra_cols
                    }
                })

                if skip_exist:
                    if "{}" in p_tmp["save_path"] and Path(p_tmp["save_path"].format(0)).exists() or \
                            Path(p_tmp["save_path"]).exists():
                        continue
                self.total_prompt_dicts.append(p_tmp)

    def get_seed(self, idx, min_val=0, max_val=1_000_000):
        # Determine seeds
        if self.seed_type == "file":
            seed = self.table.loc[idx, self.seed_col]
            if isinstance(seed, pd.Series):
                seed = int(seed.iloc[0])
            else:
                seed = int(seed)
        elif self.seed_type == "fixed":
            seed = int(self.seed)
        elif self.seed_type == "random":
            seed = random.randint(min_val, max_val)
        else:
            raise ValueError(f"Wrong seed_type: {self.seed_type}")
        return seed + self.seed_plus

    def __len__(self):
        return len(self.total_prompt_dicts)

    def __getitem__(self, index):
        return self.get_item_callback(self.total_prompt_dicts[index])


class MMUDataset(Dataset):
    def __init__(self, source, prompt=None, get_item_callback=None, save_base=None, skip_exist=False, **kwargs):
        src = decode_csv_file(source)
        if source in CAPTION_TEST_PATH:
            image_dir = Path(CAPTION_TEST_PATH[source])
        else:
            image_dir = None

        self.data = pd.read_csv(src, header=0)
        self.data.set_index("index", inplace=True)
        self.prompt = prompt
        self.get_item_callback = get_item_callback
        self.save_base = save_base
        self.skip_exist = skip_exist

        existed_ids = set()
        if self.skip_exist:
            if isinstance(self.save_base, (str, Path)) and Path(self.save_base).exists():
                saved_files = list(Path(self.save_base).glob("*.csv"))
                for file in saved_files:
                    df = pd.read_csv(file, header=0)
                    existed_ids.update(df["index"].tolist())

        # build input dict
        self.total_input_dict = []
        for id_, item in self.data.iterrows():
            if id_ in existed_ids:
                continue

            if self.prompt is not None:
                prompt = self.prompt
            elif "prompt" in item:
                prompt = item["prompt"]
            else:
                raise ValueError("No prompt found in the dataset. Please provide a prompt.")

            cache_image = image_dir / item["cache_image"] if image_dir is not None else item["cache_image"]
            input_dict = {
                "id": id_,
                "seed": item["seed"],
                "image_path": cache_image,
                "url": item["url_cos"],
                "prompt": prompt,
            }
            self.total_input_dict.append(input_dict)

    def __len__(self):
        return len(self.total_input_dict)

    def __getitem__(self, index):
        is_dummy = index // len(self) > 0
        index = index % len(self)

        data = deepcopy(self.total_input_dict[index])
        data["image"] = read_local_image(data["image_path"], apply_exif=True)
        del data["image_path"]

        data["is_dummy"] = is_dummy

        if self.get_item_callback is not None:
            return self.get_item_callback(data)
        return data


class SimpleDataset(Dataset):
    def __init__(self, source, get_item_callback=None, **kwargs):
        src = decode_csv_file(source)

        self.data = pd.read_csv(src, header=0)
        self.data.set_index("index", inplace=True)
        self.get_item_callback = get_item_callback

        # build input dict
        self.total_input_dict = []
        for id_, item in self.data.iterrows():
            prompt = item["prompt"]
            input_dict = {
                "id": id_,
                "seed": item["seed"],
                "prompt": prompt,
            }
            self.total_input_dict.append(input_dict)

    def __len__(self):
        return len(self.total_input_dict)

    def __getitem__(self, index):
        is_dummy = index // len(self) > 0
        index = index % len(self)

        data = deepcopy(self.total_input_dict[index])

        data["is_dummy"] = is_dummy

        if self.get_item_callback is not None:
            return self.get_item_callback(data)
        return data


class MessageListDataset(Dataset):
    def __init__(self, testset: str, sample_save_base, tokenizer, skip_existed=False,
                 prompt_fn=None, id_col=None, default_seed=None):
        self.tokenizer = tokenizer
        # `id_col`: when set (e.g. "fine_md5" for validation-loss sets), force this column as the row
        # index even if the CSV already has an `index` column. `default_seed`: fallback when no `seed` column.
        self.id_col = id_col
        self.default_seed = default_seed
        # Define save directory and load existing results if any
        self.save_dir = self.prepare_save_directory(testset, sample_save_base)

        # Define prompt_fn for custom manipulation of the prompt online.
        self.prompt_fn: Optional[Callable[[str, pd.Series], dict]] = prompt_fn
        if self.prompt_fn is None:
            self.prompt_fn = lambda prompt, _: {"role": "user", "content": prompt}

        (
            self.testset_type,
            self.testset,
            self.task_kwargs
        ) = self.parse_testset(testset)
        self.name_mapper = lambda x: self.task_kwargs[x] if x in self.task_kwargs else x

        self.collate_except_keys = []
        if self.testset_type == "csv":
            self.total_input_dict = self.parse_csv_dataset()
            if skip_existed:
                finished_files = list(self.save_dir.glob("results/results_*.csv"))
                if len(finished_files) > 0:
                    finished_indices = set()
                    for file in finished_files:
                        try:
                            df = pd.read_csv(file, header=0)
                            if "index" not in df.columns:
                                logger.warning(
                                    f"Result file missing 'index' column (skipping): {file}, "
                                    f"columns={list(df.columns)}"
                                )
                                continue
                            finished_indices.update(df["index"].tolist())
                        except Exception as exc:
                            logger.warning(f"Failed to read result file (skipping): {file}: {exc}")
                            continue
                    self.total_input_dict = [
                        item for item in self.total_input_dict if item["index"] not in finished_indices
                    ]
                    logger.info(
                        f"Skipped {len(finished_indices)} finished samples, {len(self.total_input_dict)} remaining."
                    )
        elif self.testset_type == "dataset":
            self.total_input_dict = self.parse_dataset()

    @staticmethod
    def prepare_save_directory(testset, sample_save_base):
        testset_renamed, *extra = testset.split("@@")
        if len(extra) > 0:
            suffix = "__" + "_".join([part.split("=")[1] for part in extra])
        else:
            suffix = ""
        sample_save_base = Path(sample_save_base)
        save_base = sample_save_base / (Path(testset_renamed).stem + suffix)
        return save_base.resolve().absolute()

    def parse_testset(self, testset):
        kwargs = {}
        if "@@" in testset:
            testset, *extra = testset.split("@@")
            for part in extra:
                key, value = part.split("=")
                kwargs[key] = value

        if testset in DATASETS and kwargs.get("metric"):
            self.dataset = load_dataset(testset, tokenizer=self.tokenizer)
            testset_type = "dataset"

        else:
            if Path(testset).exists():
                self.testset_file = testset
            else:
                self.testset_file = decode_csv_file(testset)
            testset_type = "csv"

        return testset_type, testset, kwargs

    @staticmethod
    def format_file_path(file_path: Optional[str]):
        if file_path is None:
            return None
        assert isinstance(file_path, str), f"file_path must be str, but got {type(file_path)}"

        file_path = file_path.strip()
        if file_path == "" or file_path.startswith("/") or file_path.startswith("http"):
            return file_path

        # If relative path, prepend the ASSETS_BASE path.
        file_path = Path(ASSETS_BASE) / file_path
        assert file_path.exists(), f"{file_path} does not exist"
        return str(file_path)

    @staticmethod
    def parse_media_paths(value):
        if value is None:
            return []
        if not isinstance(value, (list, tuple)) and pd.isna(value):
            return []

        if isinstance(value, str):
            raw = value.strip()
            if not raw or raw.lower() == "nan":
                return []
            if raw.startswith("["):
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    return [raw]
            else:
                return [raw]

        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if isinstance(v, str) and str(v).strip()]

        return [str(value)]

    def parse_csv_dataset(self):
        df = pd.read_csv(self.testset_file)
        if '_first_n_' in self.task_kwargs:
            df = df.head(int(self.task_kwargs['_first_n_']))
        elif '_last_n_' in self.task_kwargs:
            df = df.tail(int(self.task_kwargs['_last_n_']))

        if self.id_col is not None:
            # Validation-loss datasets force a stable unique-id column (e.g. fine_md5) as the row index,
            # overriding any existing `index` column, and fall back to `default_seed` when `seed` is absent.
            assert self.id_col in df.columns, \
                f"id_col '{self.id_col}' not found in CSV columns: {list(df.columns)}"
            df["index"] = df[self.id_col].astype(str)
            if "seed" not in df.columns:
                df["seed"] = self.default_seed if self.default_seed is not None else 42
        else:
            assert "index" in df.columns, "CSV dataset must contain 'index' column."
            assert "seed" in df.columns, "CSV dataset must contain 'seed' column."

        df = df.map(lambda x: x.format(ASSETS_BASE=ASSETS_BASE) if isinstance(x, str) and x.startswith('{ASSETS_BASE}') else x)

        if (message_col := self.name_mapper("message_list")) in df.columns:
            # OpenAI-format message list.
            def _load_message_list(row):
                resolved = []
                for msg in json.loads(row[message_col]):
                    content = msg.get("content")
                    if isinstance(content, str):
                        resolved.append(self.prompt_fn(content, row))
                    else:
                        for part in content or []:
                            for media_key in ("image", "video"):
                                if isinstance(part.get(media_key), str):
                                    part[media_key] = self.format_file_path(part[media_key])
                        resolved.append(msg)
                return resolved

            df["message_list"] = df.apply(_load_message_list, axis=1)

        elif (prompt_col := self.name_mapper("prompt")) in df.columns and \
            ((src_col := self.name_mapper("src_img_path")) in df.columns or (count_col := self.name_mapper("count")) in df.columns):

            src_col = self.name_mapper("src_img_path")
            count_col = self.name_mapper("count")
            if count_col not in df.columns:
                df["count"] = [1] * len(df)
                count_col = "count"

            def _build_messages_with_source_images(row):
                image_paths = []

                if src_col in df.columns:
                    image_paths.extend(self.parse_media_paths(row[src_col]))
                else:
                    image_paths.extend(self.parse_media_paths(row.get(self.name_mapper("src_img_path_1"))))

                count = int(row[count_col]) if row[count_col] else 1
                for i in range(1, count):
                    col_name = self.name_mapper(f"src_img_path_{i+1}")
                    if col_name in df.columns:
                        image_paths.extend(self.parse_media_paths(row.get(col_name)))

                image_paths = [p for p in image_paths if p]
                assert len(image_paths) > 0, f"No valid source image path found for index={row['index']}"

                return [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": self.format_file_path(path)}],
                    }
                    for path in image_paths
                ] + [
                    self.prompt_fn(row[prompt_col], row),
                ]

            df["message_list"] = df.apply(_build_messages_with_source_images, axis=1)

        elif (prompt_col := self.name_mapper("prompt")) in df.columns:
            # Build per-row so that a single bad caption (e.g. caption_aug raising "caption has repeat")
            # only drops that sample instead of aborting the whole dataset build.
            def _safe_build_message_list(row):
                try:
                    return [self.prompt_fn(row[prompt_col], row)]
                except Exception as exc:
                    logger.warning(
                        f"[MessageListDataset] skipping row (prompt/caption build failed): {exc}"
                    )
                    return None

            df["message_list"] = df.apply(_safe_build_message_list, axis=1)
            n_before = len(df)
            df = df[df["message_list"].notna()].reset_index(drop=True)
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                logger.warning(
                    f"[MessageListDataset] dropped {n_dropped}/{n_before} rows whose prompt/caption failed to build."
                )

        else:
            # Most commonly this means the requested prompt column (e.g. from @@prompt=/@@prompt_zh=/@@prompt_en=)
            # does not exist in this CSV. Report it explicitly so the mismatch is obvious.
            requested_prompt_col = self.name_mapper("prompt")
            raise NotImplementedError(
                f"[MessageListDataset] Cannot build message_list: requested prompt column "
                f"'{requested_prompt_col}' not found. Available columns: {list(df.columns)}."
            )
        
        # Dummy interleaved mode used for multi-image generation when boi prediction is not enabled
        if "dummy_interleaved_text_segments" in df.columns or "dummy_interleaved_num_images" in df.columns:
            def _parse_dummy_seg_cell(cell):
                if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                    return None
                if isinstance(cell, list):
                    return cell
                s = str(cell).strip()
                if s == "":
                    return None
                return json.loads(s)

            def _parse_dummy_n_cell(cell):
                if cell is None or (isinstance(cell, str) and str(cell).strip() == ""):
                    return None
                if isinstance(cell, float) and pd.isna(cell):
                    return None
                return int(cell)

            def _merge_dummy_interleaved_row(segs, n):
                seg_list = None if segs is None else list(segs)
                if n is not None and not (isinstance(n, float) and pd.isna(n)):
                    n = int(n)
                else:
                    n = None
                if seg_list is None and n is None:
                    return None
                if n is not None and n <= 0:
                    raise ValueError("`dummy_interleaved_num_images` must be positive when set.")
                seg_list = [] if seg_list is None else seg_list
                if n is not None:
                    if len(seg_list) < n:
                        seg_list = seg_list + [""] * (n - len(seg_list))
                    else:
                        seg_list = seg_list[:n]
                if len(seg_list) == 0:
                    raise ValueError(
                        "dummy interleaved: provide non-empty `dummy_interleaved_text_segments`, "
                        "or set `dummy_interleaved_num_images` > 0 (optionally with partial segments)."
                    )
                return seg_list

            if "dummy_interleaved_text_segments" in df.columns:
                seg_series = df["dummy_interleaved_text_segments"].map(_parse_dummy_seg_cell)
            else:
                seg_series = None

            if "dummy_interleaved_num_images" in df.columns:
                n_series = df["dummy_interleaved_num_images"].map(_parse_dummy_n_cell)
            else:
                n_series = None

            def _apply_merge(i):
                segs = seg_series.iloc[i] if seg_series is not None else None
                n = n_series.iloc[i] if n_series is not None else None
                return _merge_dummy_interleaved_row(segs, n)

            df["dummy_interleaved_text_segments"] = [_apply_merge(i) for i in range(len(df))]
            if "dummy_interleaved_num_images" in df.columns:
                df = df.drop(columns=["dummy_interleaved_num_images"])

        self.collate_except_keys = list(set(df.columns) - {"index", "seed", "prompt"})

        return df.to_dict(orient="records")

    def parse_dataset(self):
        data = []
        for i in range(len(self.dataset)):
            src_item = self.dataset[i]
            data_item = dict(
                index=src_item["id"],
                seed=src_item["seed"],
                prompt=src_item["input"],  # only for metric evaluation
                message_list=[
                    {"role": "user", "content": src_item["input"]}
                ],
            )
            assert "message_list" not in src_item, \
                "Key conflict: dataset item already contains 'message_list' key."
            remain_keys = [k for k in src_item.keys() if k not in ["id", "seed", "input"]]
            for k in remain_keys:
                if k in ["is_dummy"]:
                    # The `is_dummy` in dataset item is for backward compatibility, we skip it here
                    # We will add `is_dummy` flag in __getitem__.
                    continue
                data_item[k] = src_item[k]
            data.append(data_item)

        self.collate_except_keys = list(set(data[0].keys()) - {"index", "seed", "prompt"})

        return data

    def __len__(self):
        return len(self.total_input_dict)

    def __getitem__(self, index):
        is_dummy = index // len(self) > 0
        index = index % len(self)

        data = deepcopy(self.total_input_dict[index])
        data["is_dummy"] = is_dummy
        return data

    def collate_fn(self, batch):
        return except_collate_fn(batch, except_keys=self.collate_except_keys)
