import json
import os
import argparse
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from api.cos import p_upload_list
from hymm.constants import VLM_SCORE_DATA_PATH
from hymm.metrics.vlm_score.image_text_alignment_scoring import image_text_alignment_scoring, HOST
from hymm.metrics.vlm_score.scorer import ScorerFactory
from hymm.metrics.vlm_score.priors import semantic_keys, semantic_keys_priority
from hymm.metrics.vlm_score.generate_html import generate_html

# 中文字段名到英文字段名的映射
SEMANTIC_KEY_MAPPING = {
    "主要主体-名词": "primary_subject_noun",
    "主要主体-关键属性": "primary_subject_key_attributes",
    "主要主体-其他属性": "primary_subject_other_attributes",
    "主要主体-动作": "primary_subject_action",
    "次要主体-名词": "secondary_subject_noun",
    "次要主体-属性": "secondary_subject_attributes",
    "次要主体-动作": "secondary_subject_action",
    "场景-名词": "scene_noun",
    "场景-属性": "scene_attributes",
    "镜头": "shot",
    "风格": "style",
    "构图": "composition",
}


# =========================
# 1. upload images to cos
# =========================

def upload_images(image_dir, model_name, save_path):
    today = datetime.now().strftime("%Y%m%d")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        dst_data = pd.read_csv(save_path, header=0)
        print(f"Found {len(dst_data)} files in {save_path}, skip upload")
    else:
        dst_data = None

    file_list = list(Path(image_dir).glob("*.png"))
    print(f"Total {len(file_list)} files found in {image_dir}")

    if dst_data is not None:
        if len(dst_data) == len(file_list):
            print(f"All files already uploaded, skip upload")
            return dst_data
        else:
            existed = set(dst_data['img_path'].to_list())
            file_list = [file_path for file_path in file_list if str(file_path) not in existed]
            print(f"There are {len(file_list)} remaining files to upload")

    data = []
    for file_path in tqdm(file_list):
        data.append({
            'index': int(file_path.stem.split('_')[0]),       # Make sure the file name's first part is the index.
            'img_path': str(file_path),
            'key': f'jarvizhang/metrics/vlm_score/{today}_{model_name}/{file_path.name}'
        })
    data = pd.DataFrame(data)
    print(data)
    # upload
    cur_dst_data = p_upload_list(data, "img_path", "key", max_workers=32)
    # Maybe merge
    if dst_data is not None:
        dst_data = pd.concat([dst_data, cur_dst_data], ignore_index=True)
    else:
        dst_data = cur_dst_data
    dst_data.to_csv(save_path, index=False)
    return dst_data


# ===================
# 2. Make json list
# ===================

def make_json_list(testset, model_name, image_dir, work_dir, force=False):
    # Sanity checks
    ori_json_dir = Path(VLM_SCORE_DATA_PATH) / testset
    assert ori_json_dir.exists(), f"Original JSON directory {ori_json_dir} does not exist"
    ori_json_files = sorted(list(ori_json_dir.glob("*.json")))
    assert len(ori_json_files) > 0, f"No JSON files found in {ori_json_dir}"
    print(f"Found {len(ori_json_files)} JSON files in {ori_json_dir}")

    # Read for count and check if images are available
    desired_count = 0
    for ori_json_file in tqdm(ori_json_files):
        with ori_json_file.open('r') as fid:
            json_lst = json.load(fid)
            desired_count += len(json_lst)

    images_count = len(list(Path(image_dir).glob("*.png")))
    if not force and desired_count != images_count:
        raise ValueError(
            f"Number of images ({images_count}) does not match the number of prompts in JSON files ({desired_count}). "
            f"Please check the image directory {image_dir} and the JSON files in {ori_json_dir}."
        )

    # Prepare working directory
    save_dir = Path(work_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_save_path = save_dir / "image_info.csv"
    qa_save_dir = save_dir / "qa"
    qa_save_dir.mkdir(parents=True, exist_ok=True)

    # Upload images to COS
    df = upload_images(
        image_dir=image_dir,
        model_name=model_name,
        save_path=csv_save_path,
    )
    df.set_index('index', inplace=True)

    # Merge multiple json files into a single list.
    # And fill in the image info in the original JSON files
    print(f"Filling in image info for {len(ori_json_files)} JSON files...")
    save_file = qa_save_dir / "qa_with_image_info.json"
    if save_file.exists():
        with save_file.open('r') as fid:
            qa_json_list = json.load(fid)
    else:
        qa_json_list = []
        for ori_json_file in tqdm(ori_json_files):
            with ori_json_file.open('r') as fid:
                json_lst = json.load(fid)
            for item in json_lst:
                prompt_idx = item["prompt_idx"]
                # Get the image info from the DataFrame
                if force and prompt_idx not in df.index:
                    continue
                df_row = df.loc[prompt_idx]
                if "img_info_list" not in item:
                    item["img_info_list"] = []
                # Write the image info into the json_lst
                item["img_info_list"].append(dict(
                    model_name=model_name,
                    local_image_path=df_row['img_path'],
                    url_cos=df_row['url_cos'],
                ))
                qa_json_list.append(item)
    # Save the modified json_lst
    with save_file.open('w') as fid:
        json.dump(qa_json_list, fid, indent=4, ensure_ascii=False)

    return qa_json_list


# ======================
# 4. Calculate scores
# ======================

def load_json_list(res_dir, exclude=None):
    res_files = list(Path(res_dir).glob("*.json"))
    if exclude is None:
        exclude = []

    all_item_list = []
    for pth in tqdm(res_files):
        with pth.open('r') as fid:
            all_item_list.extend(json.load(fid))

    all_item_list = [item for item in all_item_list if item['prompt_idx'] not in exclude]
    return all_item_list


def image_level_score(args, res_dir, exclude=None):
    all_item_list = load_json_list(res_dir, exclude=exclude)
    print(f"Found {len(all_item_list)} items in {res_dir}")
    model_to_score_list = defaultdict(list)

    # 无优先级均分计算
    for item in tqdm(all_item_list):
        idx = item['prompt_idx']
        if idx in exclude:
            continue
        scores = item['masked_equally_weighted_score']
        img_info_list = item['img_info_list']
        assert len(scores) == len(img_info_list)
        for i, img_info in enumerate(img_info_list):
            model_name = img_info['model_name']
            model_to_score_list[model_name].append(scores[i])

    results = {}
    for model_name, score_list in model_to_score_list.items():
        rectified_score_list = [e for e in score_list if not np.isnan(e)]
        mean_score = sum(rectified_score_list) / len(rectified_score_list) if rectified_score_list else 0.0
        print()
        logger.info(f"    Model name: {model_name}")
        logger.info(f"    Mean score: {mean_score:.4f}")
        logger.info(f" Valid samples: {len(rectified_score_list)}")
        print()
        results[model_name] = {
            'mean_score': mean_score,
            'valid_samples': len(rectified_score_list)
        }
    
    return results


def semantic_level_score(args, res_dir, exclude=None):
    item_list = load_json_list(res_dir, exclude=exclude)
    scorer = ScorerFactory.create_scorer(args.score_weighting_scheme)
    results = scorer.compute_scores(item_list)

    output = []
    for result in results:
        print(result.keys())
        stats = result["stats"]
        mean_score = stats.get('image_level_accuracy', 0.0)    # bad name (`image_level` actually is `semantic_level`), but kept for compatibility
        image_root = result["image_root"]
        model_name = result['model_name'] if 'model_name' in result else Path(image_root).name
        print()
        logger.info(f"    Model name: {model_name}")
        logger.info(f"    Mean score: {mean_score:.4f}")
        print()
        
        # 提取各个语义维度的分数
        semantic_scores = {}
        for key in semantic_keys:
            semantic_scores[key] = stats[key]['accuracy'] if isinstance(stats[key], dict) else stats[key]
        
        output.append({
            'model_name': model_name,
            'mean_score': mean_score,
            'semantic_scores': semantic_scores,
            'image_root': image_root,
        })

    markdown_field_level_accuracy = [
        "| Image | " + " | ".join(semantic_keys) + " |",
        "|---|" + "---|" * len(semantic_keys),
        "|   | " + " | ".join(semantic_keys_priority) + " |"
    ]
    for result in results:
        model_name = result['model_name'] if 'model_name' in result else Path(result["image_root"]).name
        stats = result["stats"]
        acc_list = [f"{stats[k]['accuracy']:.4f}" for k in semantic_keys]
        markdown_field_level_accuracy.append(f"| {model_name} | " + " | ".join(acc_list) + " |")
    print("\n".join(markdown_field_level_accuracy))
    
    return output


def safe_save_json(data, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = save_path.parent / (save_path.name + ".tmp")
    with temp_path.open('w') as fid:
        json.dump(data, fid, ensure_ascii=False, indent=4)
    if save_path.exists():
        save_path.unlink()
    temp_path.rename(save_path)


def parse_step_from_model_name(model_name: str):
    """从模型名称中解析迭代步数，如 torch_iter_0011000_zh_xxx -> 11000"""
    m = re.search(r"iter_(\d+)", model_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def extract_model_name_from_path(res_dir):
    """从res_dir路径中提取模型名称"""
    # 路径格式: __eval/vlm_score/{testset}/{model_name}/res_{model_marker}
    path = Path(res_dir)
    if path.name.startswith('res_'):
        # 父目录就是模型名称
        model_name = path.parent.name
    else:
        # 尝试从路径中提取
        parts = path.parts
        if 'vlm_score' in parts:
            idx = parts.index('vlm_score')
            if idx + 1 < len(parts):
                model_name = parts[idx + 1]
            else:
                model_name = path.name
        else:
            model_name = path.name
    
    return model_name


def save_metric_results_for_model(
    result: dict,
    image_dir: Path,
    testset: str,
    count: int,
    model_marker: str = None,
):
    """
    将每项指标结果追加保存到对应的 metric json 文件中。

    目录结构：
        如果 image_dir 是: <...>/iter_000xxx/<testset>/images
        保存到: <...>/iter_000xxx/<testset>/metric_results_{model_marker}/<metric_name>.json

    每个 json 文件的格式：
    [
        {
            "timestamp": "20260220-043253",
            "metric": "image_level_score",
            "testset": "internal_arena_test_long_long_all_v2__prompt_zh_long_long",
            "value": xxx,
            "count": 500,
            "step": 11000
        },
        ...
    ]
    """
    # image_dir 是图像目录（例如 .../images），取其父目录作为 testset 目录
    # 然后在 testset 目录下创建 metric_results_{model_marker} 目录
    # image_dir 应该已经是 Path 对象（在调用前已转换）
    testset_dir = image_dir.parent
    if model_marker:
        # 将 model_marker 中的特殊字符替换为下划线，用于目录名
        model_marker_safe = model_marker.replace("-", "_").replace(".", "_")
        metric_dir = testset_dir / f"metric_results_{model_marker_safe}"
    else:
        metric_dir = testset_dir / "metric_results"
    metric_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metric_testset = testset
    step = parse_step_from_model_name(result.get("model_name", ""))

    # 将result保存结果到txt文件
    with open(metric_dir / "result.txt", "w") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

    # 需要写入的所有指标（除去 model_name）
    metrics_items = {
        k: v for k, v in result.items()
        if k != "model_name"
    }

    for metric_name, value in metrics_items.items():
        metric_file = metric_dir / f"{metric_name}.json"

        # 如果该 metric 已经有记录，并且已经包含当前 step & testset，则跳过
        existing = []
        if metric_file.exists():
            try:
                with metric_file.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        if step is not None and any(
            (item.get("testset") == metric_testset and item.get("step") == step)
            for item in existing
        ):
            # 已经有当前 step 的结果了，跳过
            logger.info(f"[metric_results] {metric_name} for testset={metric_testset}, step={step} already exists. skip.")
            continue

        entry = {
            "timestamp": timestamp,
            "metric": metric_name,
            "testset": metric_testset,
            "value": float(value),
            "count": int(count),
        }
        if step is not None:
            entry["step"] = step

        existing.append(entry)
        with metric_file.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)


def calc_final_score(args, res_dir, save_results=True, testset=None, image_dir=None, model_marker=None):
    logger.info(f"Calculating final score from results in {res_dir}")
    res_dir = Path(res_dir)
    
    # 提取模型名称
    model_name = extract_model_name_from_path(res_dir)
    
    # 确保exclude不为None
    exclude = args.exclude if hasattr(args, 'exclude') and args.exclude else []
    
    image_level_results = None
    semantic_level_results = None
    
    for score_type in args.score_type:
        print()
        logger.info("=" * 40)
        logger.info(f"    Score type: {score_type}")
        logger.info("=" * 40)

        if score_type == "image_level":
            image_level_results = image_level_score(args, res_dir, exclude)
        elif score_type == "semantic_level":
            semantic_level_results = semantic_level_score(args, res_dir, exclude)
    
    # 保存结果到JSON文件
    if save_results:
        result = {'model_name': model_name}

        # 添加图像级别分数
        if image_level_results:
            # image_level_results是一个字典，key是model_name
            model_name_from_image = list(image_level_results.keys())[0]
            image_level_data = image_level_results[model_name_from_image]
            result['image_level_score'] = image_level_data['mean_score']
            # 确保模型名称一致
            if model_name_from_image != model_name:
                model_name = model_name_from_image
                result['model_name'] = model_name
        
        # 添加语义级别分数
        if semantic_level_results:
            semantic_level_data = semantic_level_results[0]
            result['semantic_level_score'] = semantic_level_data['mean_score']
            # 确保模型名称一致
            if semantic_level_data['model_name'] != model_name:
                model_name = semantic_level_data['model_name']
                result['model_name'] = model_name
            
            # 添加12个语义维度分数（使用英文字段名）
            for key in semantic_keys:
                english_key = SEMANTIC_KEY_MAPPING.get(key, key.lower().replace('-', '_').replace(' ', '_'))
                result[english_key] = semantic_level_data['semantic_scores'].get(key, 0.0)
        
        # 只有当有结果时才保存
        if len(result) > 1:  # 除了model_name之外还有其他字段
            # 如果提供了testset或image_dir，保存到metric_results目录
            # 优先使用传入的image_dir，如果没有则尝试从结果中获取
            if image_dir is None and semantic_level_results:
                # 从结果中获取image_root作为image_dir
                image_dir = semantic_level_results[0].get("image_root")
                if image_dir:
                    logger.info(f"Using image_dir from results: {image_dir}")
            
            if image_dir:
                # 确保 image_dir 是 Path 对象
                image_dir = Path(image_dir)
                # 如果testset未提供，从image_dir路径中提取（image_dir的父目录名称）
                if testset is None:
                    testset = image_dir.parent.name
                    logger.info(f"Extracted testset from image_dir: {testset}")
                
                count = image_level_results[list(image_level_results.keys())[0]]['valid_samples'] if image_level_results else len(load_json_list(res_dir, exclude))
                save_metric_results_for_model(
                    result=result,
                    image_dir=image_dir,
                    testset=testset,
                    count=count,
                    model_marker=model_marker,
                )
                metric_dir_name = f"metric_results_{model_marker.replace('-', '_').replace('.', '_')}" if model_marker else "metric_results"
                logger.info(f"Metric results saved to {metric_dir_name} directory")
            elif testset:
                logger.warning("image_dir not provided and cannot be inferred from results, skipping metric_results save")


# ======================
#   5. Visualize
# ======================

def visualize(res_dirs, res_tags, save_path):
    assert len(res_dirs) == len(res_tags), "res_dirs and res_tags must have the same length"
    for res_dir in res_dirs:
        logger.info(f"Visualizing {res_dir}")
    generate_html(res_dirs, res_tags, save_path)


# ======================
#       Pipeline
# ======================

def vlm_score(args, testset, model_name, image_dir, work_dir=None, model_marker='api_google_gemini-2.5-pro',
              force=False, num_workers=32, api_name="data_eval", hosts=None):
    """
    Run VLM Score for a testset and corresponding images.

    Parameters
    ----------
    args: argparse.Namespace
        Command line arguments parsed by argparse.
    testset: str
        testset name. Currently only supports ["internal_arena_500_sem_split"].
    model_name: str
        A custom model name for the data. Used to define the saving directory.
    image_dir: str
        A directory containing images to be scored. The images should be named as "{index}.png" or "{index}_*.png",
        where index is the index of the image in the testset.
    work_dir: str, optional
        A directory to save the middle results and final results. If not provided, will be set to image_dir.parent.
    model_marker: str
        The model marker to select the model for judging.
    force: bool
        Whether to force run even if the number of images does not match the number of prompts in the testset.
    num_workers: int
        Number of workers for local inference.
    api_name: str
        Name of api to use. Options are ["data_eval", "chat_completions"]
    hosts: str
        Custom hosts for the API.
    """
    image_dir = Path(image_dir)
    # 如果 work_dir 未提供，则使用 image_dir 的父目录
    if work_dir is None:
        work_dir = image_dir.parent
    model_marker_repr = model_marker.replace("-", "_").replace(".", "_")
    work_dir = Path(work_dir)
    save_dir = work_dir / model_name / f"res_{model_marker_repr}"
    save_dir.mkdir(parents=True, exist_ok=True)

    app_id = os.getenv("VLM_SCORE_APP_ID")
    token = os.getenv("VLM_SCORE_TOKEN")
    params_dict = {
        "account": app_id,
        "passwd": token,
        "num_workers": num_workers,
        "max_api_qps": min(100, num_workers),
        "model_marker": model_marker,
        "api_name": api_name,
        "host": hosts or HOST,
    }
    print(f'Adopt {model_marker} for inference...')

    qa_json_list = make_json_list(testset, model_name, image_dir, work_dir, force)

    # Find finished records
    save_file = save_dir / f"res.json"
    if save_file.exists():
        with save_file.open('r') as f:
            res_list = json.load(f)
        print(f"Found {len(res_list)} finished records in {save_file}")
        existed_ids = {item['prompt_idx'] for item in res_list}
        qa_json_list = [item for item in qa_json_list if item['prompt_idx'] not in existed_ids]
    else:
        res_list = []

    print(f"Total {len(qa_json_list)} records to process")
    print(f"Start processing by VLM...")
    batches = []
    batch_size = 100
    for i in range(0, len(qa_json_list), batch_size):
        batches.append(qa_json_list[i:i + batch_size])
    pbar = tqdm(enumerate(batches), total=len(batches))
    for i, batch_qa_json in pbar:
        item_list = image_text_alignment_scoring(batch_qa_json, **params_dict)
        res_list.extend(item_list)
        safe_save_json(res_list, save_file)

    # Final save
    safe_save_json(res_list, save_file)
    print(f"Saved JSON to {save_file}")
    # Calculate final score
    calc_final_score(args, save_dir, save_results=True, testset=testset, image_dir=image_dir, model_marker=model_marker)
    # Generate visualization
    if args.save:
        save_path = Path(args.save)
    else:
        save_path = work_dir / model_name / f"{model_name}.html"
    visualize([str(save_dir)], [None], str(save_path))


def merge_jsons(json_dir_list, save_file, model_name):
    # 只支持单个模型的结果合并
    json_list_list = []
    for res_dir in json_dir_list:
        all_item_list = load_json_list(res_dir)
        json_list_list.append(all_item_list)

    # 确保所有的 json_list 的长度和顺序都一样
    base_len = len(json_list_list[0])
    for i, json_list in enumerate(json_list_list):
        assert len(json_list) == base_len, \
            f"All JSON lists must have the same length, but got {len(json_list)} and {base_len} for index {i}"

    merged_list = []
    for items in zip(*json_list_list):
        merged_item = deepcopy(items[0])
        # 修改 image_info_list 的 model_name
        merged_item["img_info_list"][0]["model_name"] = model_name
        max_index = np.argmax([item["masked_equally_weighted_score"][0] for item in items])
        merged_item["img_info_list"][0]["local_image_path"] = items[max_index]["img_info_list"][0]["local_image_path"]
        merged_item["img_info_list"][0]["url_cos"] = items[max_index]["img_info_list"][0]["url_cos"]
        merged_item["masked_equally_weighted_score"] = [items[max_index]["masked_equally_weighted_score"][0]]
        merged_item["score_certainty"] = max([item["score_certainty"] for item in items])
        merged_item["exact_scoring_points_mask"] = {
            key: any([item["exact_scoring_points_mask"][key] for item in items])
            for key in items[0]["exact_scoring_points_mask"].keys()
        }
        # merged_item["structured_semantic_points_matching"] = [{
        #     key: [
        #         max([item["structured_semantic_points_matching"][0][key][i] for item in items])
        #         for i in range(len(items[0]["structured_semantic_points_matching"][0][key]))
        #     ]
        #     for key in items[0]["structured_semantic_points_matching"][0].keys()
        # }]
        merged_item["structured_semantic_points_matching"] = items[max_index]["structured_semantic_points_matching"]
        merged_list.append(merged_item)

    safe_save_json(merged_list, save_file)
    print(f"Merged JSON saved to {save_file}")


def main():
    parser = argparse.ArgumentParser(description="Run VLM Score for a testset and corresponding images.")

    def add_score_args(parser_):
        parser_.add_argument("--score-type", type=str, default=['image_level', 'semantic_level'],
                             choices=['image_level', 'semantic_level'],
                             nargs='+', help="Type of score calculation.")
        parser_.add_argument('--score_weighting_scheme', type=str, default='equal_weighted')
        parser_.add_argument("--exclude", type=int, nargs="+", default=[], help="Exclude images from scoring.")
        return parser_

    def add_vis_parser(parser_):
        parser_.add_argument("--save", type=str, default="visualization.html", help="Path to save the visualization HTML file.")
        return parser_

    subparsers = parser.add_subparsers(dest="task", required=True)
    run_parser = subparsers.add_parser("run", help="Run VLM Score for a testset and corresponding images.")
    run_parser.add_argument("--testset", type=str, required=True, help="Testset name.")
    run_parser.add_argument("--model-name", type=str, required=True, help="Custom model name for the data.")
    run_parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images to be scored.")
    run_parser.add_argument("--work-dir", type=str, default=None, help="Working directory for intermediate results. If not provided, will be set to image_dir.parent.")
    run_parser.add_argument("--run-local", action='store_true', help="Run locally instead of using the API. Deprecated.")
    run_parser.add_argument("--model-marker", type=str, default='api_google_gemini-2.5-pro',)
    run_parser.add_argument("--force", action='store_true', help="Force run even if images are not complete.")
    run_parser.add_argument("--num-workers", type=int, default=32, help="Number of workers for local inference.")
    run_parser.add_argument("--api-name", type=str, default='data_eval', choices=["data_eval", "chat_completions"], help="Name of api to use.")
    run_parser.add_argument("--hosts", type=str, nargs='+', help="Custom host for the API.")
    add_score_args(run_parser)
    add_vis_parser(run_parser)

    score_parser = subparsers.add_parser("score", help="Calculate final score from the results.")
    score_parser.add_argument("data", type=str, help="Directory containing the results to score.")
    add_score_args(score_parser)

    vis_parser = subparsers.add_parser("vis", help="Visualize the results.")
    vis_parser.add_argument("data", type=str, nargs="+", help="Directories containing the results to visualize.")
    vis_parser.add_argument("--res-tags", type=str, nargs="+", help="Tags for each result directory.")
    add_vis_parser(vis_parser)

    merge_parser = subparsers.add_parser("merge", help="Merge multiple result JSON files by maximum. Used for best-of-N evaluation.")
    merge_parser.add_argument("data", type=str, nargs="+", help="List of JSON directories to merge.")
    merge_parser.add_argument("--model-name", type=str, help="Model name to set in the merged JSON file.")
    merge_parser.add_argument("--save", type=str, required=True, help="Path to save the merged JSON file.")

    args = parser.parse_args()

    if args.task == "run":
        vlm_score(
            args=args,
            testset=args.testset,
            model_name=args.model_name,
            image_dir=args.image_dir,
            work_dir=args.work_dir,
            model_marker=args.model_marker,
            force=args.force,
            num_workers=args.num_workers,
            api_name=args.api_name,
            hosts=args.hosts,
        )
    elif args.task == "score":
        calc_final_score(args, args.data)
    elif args.task == "vis":
        visualize(args.data, args.res_tags, args.save)
    elif args.task == "merge":
        merge_jsons(args.data, args.save, args.model_name)


if __name__ == '__main__':
    main()
