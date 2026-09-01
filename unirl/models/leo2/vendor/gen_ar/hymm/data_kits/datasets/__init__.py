# ================================================================
# This file is part of the package `hymm`.
# We collect all the dataset classes in this file for easy access.
# ================================================================

DATASETS = {
    "coco30k", "coco6k", "coco3k",
    "t2i_compbench",
    "dpg_bench",
    "geneval",
    "flickr30k_test", "nocaps_val",
    "vqav2_val",
    "mmmu", "mmmu_pro",
    "MMBench_DEV_EN", "MMBench_DEV_CN", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11",
    "counting_eval",
    "mmlu_bench",
    "mmlu_pro_bench",
    "imagenet",
    "audioset",
    "vggsound",
}


def load_dataset(dataset_name, **kwargs):
    if dataset_name == "coco30k":
        from .coco import COCODataset
        from ...constants import COCO30K_PATH

        dataset = COCODataset(COCO30K_PATH)

    elif dataset_name == "coco6k":
        from .coco import COCODataset
        from ...constants import COCO6K_PATH

        dataset = COCODataset(COCO6K_PATH)

    elif dataset_name == "coco3k":
        from .coco import COCODataset
        from ...constants import COCO3K_PATH

        dataset = COCODataset(COCO3K_PATH)

    elif dataset_name == "t2i_compbench":
        from .t2i_compbench import T2ICompBenchDataset
        from ...constants import T2I_COMPBENCH_PATH

        dataset = T2ICompBenchDataset(T2I_COMPBENCH_PATH)

    elif dataset_name == "dpg_bench":
        from .dpg_bench import DPGBenchDataset
        from ...constants import DPG_BENCH_DATA

        dataset = DPGBenchDataset(csvs=DPG_BENCH_DATA)

    elif dataset_name == "geneval":
        from .geneval import GenEvalDataset
        from ...constants import GENEVAL_DATA

        dataset = GenEvalDataset(jsonl=GENEVAL_DATA)

    elif dataset_name in ["flickr30k_test", "nocaps_val"]:
        from .flickr30k import CaptionTestDataset
        from ...constants import CAPTION_TEST_PATH

        dataset = CaptionTestDataset(CAPTION_TEST_PATH[dataset_name], **kwargs)

    elif dataset_name == "vqav2_val":
        from .vqa import VQADataset
        from ...constants import VQA_DATA_PATH, COCO_VAL2014

        dataset = VQADataset(test=VQA_DATA_PATH[dataset_name]["test"],
                             dataset_name=dataset_name,
                             image_base=COCO_VAL2014,
                             max_new_tokens=VQA_DATA_PATH[dataset_name]["max_new_tokens"],
                             **kwargs,
                             )

    elif dataset_name in ["mmmu", "mmmu_pro"]:
        from .mmmu import MMMUDataset
        from ...constants import MMMU_PATH

        dataset = MMMUDataset(dataset_name=dataset_name,
                              data_path=MMMU_PATH[dataset_name]["path"],
                              split=MMMU_PATH[dataset_name]["split"],
                              target_size=512)

    elif dataset_name in ["MMBench_DEV_EN", "MMBench_DEV_CN", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11"]:
        from .mmbench import MMBenchDataset
        from ...constants import LMUDataRoot

        dataset = MMBenchDataset(LMUDataRoot=LMUDataRoot,
                                 dataset_name=dataset_name,
                                 target_size=512)

    elif dataset_name == "counting_eval":
        from .counting_eval import CountingEvalDataset
        from ...constants import COUNTING_EVAL_DATA

        dataset = CountingEvalDataset(jsonl=COUNTING_EVAL_DATA)

    elif dataset_name == "mmlu_bench":
        from .mmlu_bench import MMLUBenchDataset
        from ...constants import MMLU_BENCH_DATA

        dataset = MMLUBenchDataset(MMLU_BENCH_DATA, **kwargs)

    elif dataset_name == "mmlu_pro_bench":
        from .mmlupro_bench import MMLUProBenchDataset
        from ...constants import MMLU_PRO_BENCH_DATA

        dataset = MMLUProBenchDataset(MMLU_PRO_BENCH_DATA)

    elif dataset_name == "imagenet":
        from .label import LabelDataset

        labels = []
        for i in range(1000):
            labels.extend([i] * 30)
        dataset = LabelDataset(labels=labels, save_template='{}')

    elif dataset_name == "audioset":
        from .audio_parquet import ParquetDataset
        from ...constants import AUDIOSET
        parquet_file = f"{AUDIOSET}/val.parquet"
        dataset = ParquetDataset(source=parquet_file, data_root=AUDIOSET)

    elif dataset_name == "vggsound":
        from .audio_parquet import ParquetDataset
        from ...constants import VGGSOUND
        parquet_file = f"{VGGSOUND}/val.parquet"
        dataset = ParquetDataset(source=parquet_file, data_root=VGGSOUND)

    elif "chinese_woman_face" in dataset_name:
        prompts_name = dataset_name.split("chinese_woman_face_")[1]
        from .csv_folder_dataset import CSVImageFolderDataset
        from ...constants import REAL_WORLD_FACE_DATA_FOLDER
        dataset = CSVImageFolderDataset(
            source_dir=REAL_WORLD_FACE_DATA_FOLDER,
            save_template="{}",
            image_size=(512, 512),
            image_suffix=".png",
            csv_name=f"{prompts_name}.csv",
            src_folder_name="src_image",
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset
