import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from hymm.constants import CAPTION_TEST_PATH

OCRBench_score = {"Regular Text Recognition": 0, "Irregular Text Recognition": 0, "Artistic Text Recognition": 0,
                  "Handwriting Recognition": 0,
                  "Digit String Recognition": 0, "Non-Semantic Text Recognition": 0, "Scene Text-centric VQA": 0,
                  "Doc-oriented VQA": 0,
                  "Key Information Extraction": 0, "Handwritten Mathematical Expression Recognition": 0}
IMAGE_FOLDER = Path(CAPTION_TEST_PATH["mmu_ocrbench"]) / "OCRBench_Images"
JSON_FILE = Path(CAPTION_TEST_PATH["mmu_ocrbench"]) / "OCRBench.json"


def calc_OCRBench(data):
    for i in tqdm(range(len(data))):
        data_type = data[i]["type"]
        dataset_name = data[i]["dataset_name"]
        answers = data[i]["answers"]
        predict = data[i]['predict']
        data[i]['result'] = 0
        if dataset_name == "HME100k":
            if isinstance(answers, list):
                for j in range(len(answers)):
                    answer = answers[j].strip().replace("\n", " ").replace(" ", "")
                    predict = predict.strip().replace("\n", " ").replace(" ", "")
                    if answer in predict:
                        data[i]['result'] = 1
            else:
                answers = answers.strip().replace("\n", " ").replace(" ", "")
                predict = predict.strip().replace("\n", " ").replace(" ", "")
                if answers in predict:
                    data[i]['result'] = 1
        else:
            if isinstance(answers, list):
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace("\n", " ")
                    predict = predict.lower().strip().replace("\n", " ")
                    if answer in predict:
                        data[i]['result'] = 1
            else:
                answers = answers.lower().strip().replace("\n", " ")
                predict = predict.lower().strip().replace("\n", " ")
                if answers in predict:
                    data[i]['result'] = 1

    for i in range(len(data)):
        if data[i].get("result", 100) == 100:
            continue
        OCRBench_score[data[i]['type']] += data[i]['result']
    recognition_score = OCRBench_score['Regular Text Recognition'] + OCRBench_score['Irregular Text Recognition'] + \
                        OCRBench_score['Artistic Text Recognition'] + OCRBench_score['Handwriting Recognition'] + \
                        OCRBench_score['Digit String Recognition'] + OCRBench_score['Non-Semantic Text Recognition']
    Final_score = recognition_score + OCRBench_score['Scene Text-centric VQA'] + OCRBench_score['Doc-oriented VQA'] + \
                  OCRBench_score['Key Information Extraction'] + OCRBench_score[
                      'Handwritten Mathematical Expression Recognition']
    print("###########################OCRBench##############################")
    print(f"Text Recognition(Total 300):{recognition_score}")
    print("------------------Details of Recognition Score-------------------")
    print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}")
    print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}")
    print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}")
    print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}")
    print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}")
    print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}")
    print("----------------------------------------------------------------")
    print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}")
    print("----------------------------------------------------------------")
    print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}")
    print("----------------------------------------------------------------")
    print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}")
    print("----------------------------------------------------------------")
    print(
        f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}")
    print("----------------------Final Score-------------------------------")
    print(f"Final Score(Total 1000): {Final_score}")

    return OCRBench_score, Final_score


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True, help="Model prediction result directory.")
    parser.add_argument("--model-tag", type=str, required=True, help="Model tag for the results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory {result_dir} does not exist.")

    # 创建 score 的储存目录
    save_file = result_dir.parent / f"{result_dir.stem}_score.csv"

    # 读取模型预测结果
    print(f"Reading preds from {result_dir}...")
    dfs = []
    for res_file in tqdm(list(result_dir.glob("all_results.csv"))):
        df = pd.read_csv(res_file, header=0, dtype={'index': str, 'answer': str}).fillna("")
        dfs.append(df)
    df = pd.concat(dfs).set_index("index")
    assert len(df) == len(data), f"Length of data {len(data)} and prediction {len(df)} do not match."

    # 把模型预测结果写入 data
    for i in range(len(data)):
        question = data[i]['question']
        result_id = f"{data[i]['dataset_name']}_{data[i]['id']}"
        if "response" in df.columns:
            data[i]['predict'] = json.loads(df.loc[result_id, 'response'])["content"][0]["text"]
        else:
            data[i]['predict'] = df.loc[result_id, 'answer']

    # 调用官方的评估方法
    print(f"Evaluating OCRBench...")
    score_dict, score_final = calc_OCRBench(data)

    # 保存到文件
    data = pd.DataFrame([dict(
        model_tag=args.model_tag,
        score_final=score_final,
        **score_dict,
    )])
    data.to_csv(save_file, index=False)
