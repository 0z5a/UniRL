from datetime import datetime, timezone
import base64
import hmac
import hashlib
import requests
import uuid
import pandas as pd
import csv
import re
from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager, Process
from functools import partial
import os
import time
import argparse
from pathlib import Path
from loguru import logger
from api.cos import p_upload_list
from tqdm import tqdm
from omnicontext_prompt_generator import PromptGenerator
from tools.parallel_utils import parallel_auto
from json_util import mllm_output_to_dict

API_VERSION = "v2.03"

# =========================
# 1. upload images to cos
# =========================

def upload_images(model_name, save_path, image_dir=None, image_list=None, mode='tgt'):
    today = datetime.now().strftime("%Y%m%d")
    assets_base = os.getenv("ASSETS_BASE", "/apdcephfs_wza/1_public_models/hymm_ar_assets")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        dst_data = pd.read_csv(save_path, header=0)
        print(f"Found {len(dst_data)} files in {save_path}, skip upload")
    else:
        dst_data = None

    if image_list is None:
        file_list = list(Path(image_dir).glob("*.png"))
    else:
        file_list = image_list
    print(f"Total {len(file_list)} files found")

    if dst_data is not None:
        if len(dst_data) == len(file_list):
            print(f"All files already uploaded, skip upload")
            return dst_data
        else:
            existed = set(dst_data['img_path'].to_list())
            file_list = [file_path for file_path in file_list if str(file_path) not in existed]
            print(f"There are {len(file_list)} remaining files to upload")

    data = []
    for file_path in file_list:
        file_path = Path(str(file_path).replace("{ASSETS_BASE}", assets_base))
        if mode == 'tgt':
            data.append({
                'index': int(file_path.stem.split('_')[0]),       # Make sure the file name's first part is the index.
                'img_path': str(file_path),
                'key': f'jarvizhang/metrics/omnicontext/{today}_{model_name}/{file_path.name}'
            })
        elif mode == 'src':
            data.append({
                'index': int(file_path.stem.split('_')[0]),       # Make sure the file name's first part is the index.
                'img_path': str(file_path),
                'key': f'jarvizhang/metrics/omnicontext/{today}_{model_name}/{file_path.name}'
            })

    data = pd.DataFrame(data)
    print(data)
    # upload
    cur_dst_data = p_upload_list(data, "img_path", "key", max_workers=64)
    # Maybe merge
    if dst_data is not None:
        dst_data = pd.concat([dst_data, cur_dst_data], ignore_index=True)
    else:
        dst_data = cur_dst_data
    dst_data.to_csv(save_path, index=False)
    return dst_data


def get_simple_auth(source, SecretId, SecretKey):
    dateTime = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = "hmac id=\"" + SecretId + "\", algorithm=\"hmac-sha1\", headers=\"date source\", signature=\""
    signStr = "date: " + dateTime + "\n" + "source: " + source
    sign = hmac.new(SecretKey.encode(), signStr.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    sign = auth + sign + "\""
    return sign, dateTime


class Api:
    def __init__(self, host, user, apikey):
        self.host = host
        self.user = user
        self.apikey = apikey
        self.timeout = 3600  # 超时时间

    def get_header(self):
        source = 'xxxxxxx'  # 签名水印值，可填写任意值
        sign, dateTime = get_simple_auth(source, self.user, self.apikey)
        headers = {'Apiversion': API_VERSION, 'Authorization': sign, 'Date': dateTime, 'Source': source}
        return headers

    def call_data_eval(self, prompt, image_url_list):
        base_url = self.host + '/api/v1/data_eval'
        data = {
            "request_id": str(uuid.uuid4()),
            "model_marker": "api_google_gemini-2.5-pro",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "value": prompt
                        },
                        # {
                        #     "type": "image_url",
                        #     "value": img1_url
                        # },
                        # {
                        #     "type": "image_url",
                        #     "value": img2_url
                        # }
                    ]
                }
            ],
            "params": {},
            "timeout": 6000
        }

        for img_url in image_url_list:
            data['messages'][0]['content'].append({"type": "image_url", "value": img_url})

        # print(data)
        headers = dict(self.get_header())
        rsp = requests.post(url=base_url, headers=headers, json=data, timeout=self.timeout)
        return rsp


def process_row(row, api):
    instruction = row['prompt']
    with_scene = row['with_scene']
    image_url_list = [row[f'src_img_cos_{i}'] for i in [1,2,3] if not pd.isna(row[f'src_img_cos_{i}'])] + [row["tgt_img_cos"]]

    max_retry = 100
    PF_score, SC_score = None, None
    for i in range(max_retry):
        ans = None
        try:
            pg = PromptGenerator()
            PF_prompt = pg(instruction, task_type="prompt_following")
            SC_prompt = pg(instruction, task_type="subject_consistency", with_scene=with_scene)

            ret = api.call_data_eval(PF_prompt, image_url_list)
            ans = ret.json()
            PF_res = ans["answer"][0]["value"]
            PF_score = mllm_output_to_dict(PF_res)

            ret = api.call_data_eval(SC_prompt, image_url_list)
            ans = ret.json()
            SC_res = ans["answer"][0]["value"]
            SC_score = mllm_output_to_dict(SC_res)

            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(ans if ans is not None else f"{e.__class__.__name__}: {e}\nInstruction: {instruction}")
            if i < max_retry - 1:
                time.sleep(0.5)

    info = {
        **row[['index', 'prompt', 'src_img_cos_1', 'src_img_cos_2', 'src_img_cos_3', 'tgt_img_cos']],
        'PF_score': PF_score and PF_score['score'] or -1,
        'SC_score': SC_score and SC_score['score'] or -1,
        'PF_score_reasoning': PF_score and PF_score['reasoning'] or 'error',
        'SC_score_reasoning': SC_score and SC_score['reasoning'] or 'error',
    }
    return info


def call_gemini_sc(df, output_csv, num_workers=128):
    HOST = "trpc-gpt-eval.production.polaris"
    app_id = os.getenv("VLM_SCORE_APP_ID")
    token = os.getenv("VLM_SCORE_TOKEN")
    if app_id is None or token is None:
        raise ValueError("Please set the environment variables VLM_SCORE_APP_ID and VLM_SCORE_TOKEN.")
    api = Api("http://{}:8080".format(HOST), app_id, token)

    job_list = [[df.loc[idx], api] for idx in df.index]
    parallel_auto(output_csv, num_workers, 'call_gemini_sc')(process_row)(job_list)

    return pd.read_csv(output_csv, header=0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True, help="Directory with AI generated images")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--testset", type=str, default="data/test/editing_omnicontext.csv", help="Testset csv file")
    parser.add_argument("--work-dir", type=str, default="__eval/omnicontext_score/editing_omnicontext", help="Working directory")
    parser.add_argument("--only-calc-score", action='store_true', help="Only calculate score")
    parser.add_argument("--with-scene", action='store_true', default=False)
    parser.add_argument("--only-multi-ref", action='store_true', help="Only multi reference image")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # 创建工作目录
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_file = work_dir / args.model_name / "pf_sc_score.csv"

    # 读取图像目录
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")
    image_files = list(image_dir.glob("*.png"))
    if len(image_files) == 0:
        raise ValueError(f"No PNG images found in the directory {image_dir}.")
    id2file = {int(f.stem): f for f in image_files}

    # 读取测试集并校验
    test_df = pd.read_csv(args.testset, header=0)
    if len(image_files) < len(test_df):
        logger.warning(f"The number of image files {len(image_files)} is less than the number of "
                       f"records in the CSV file {len(test_df)}. ")

    # 上传图像到 COS, 并保存 cos_url 到 image_info 文件. 会自动跳过已完成的.
    image_info_save_path = work_dir / args.model_name / "image_info.csv"
    res_df = upload_images(
        image_dir=image_dir,
        model_name=args.model_name,
        save_path=image_info_save_path,
    )
    id2url = res_df.set_index('index')['url_cos'].to_dict()
    test_df["tgt_img_path"] = test_df['index'].apply(lambda x: id2file.get(x, None))
    test_df["tgt_img_cos"] = test_df['index'].apply(lambda x: id2url.get(x, None))

    for i in [1, 2, 3]:
        image_info_save_path = work_dir / args.model_name / f"src_{i}_image_info.csv"
        path_objs = [Path(p) for p in test_df[f'src_img_path_{i}'].dropna().tolist()]
        res_df = upload_images(
            model_name=args.model_name,
            save_path=image_info_save_path,
            image_list=path_objs,
            mode='src'
        )
        id2url = res_df.set_index('index')['url_cos'].to_dict()
        test_df[f"src_img_cos_{i}"] = test_df['index'].apply(lambda x: id2url.get(x, None))

    test_df = test_df.dropna(subset=['tgt_img_cos', 'prompt'])
    test_df['with_scene'] = [args.with_scene] * len(test_df)
    if args.only_multi_ref:
        test_df = test_df[test_df['count'] >= 2]
    # test_df = test_df[:2]
    print(f"Total {len(test_df)} records after dropping NaN values.")
    if args.only_calc_score:
        score_df = pd.read_csv(output_file, header=0)
    else:
        score_df = call_gemini_sc(test_df, output_file, num_workers=32)

    avg_PF_score = score_df['PF_score'].mean()
    avg_SC_score = score_df['SC_score'].mean()
    valid_num_records = len(score_df[(score_df['PF_score'] >= 0) & (score_df['SC_score'] >= 0)])
    valid_num_records_PF = len(score_df[score_df['PF_score'] >= 0])
    not_valid_num_records_PF = len(score_df[score_df['PF_score'] < 0])
    valid_num_records_SC = len(score_df[score_df['SC_score'] >= 0])
    not_valid_num_records_SC = len(score_df[score_df['SC_score'] < 0])
    print(f"==========================")
    print(f"Model name: {args.model_name}")
    print(f"Valid number of records: {valid_num_records}")
    print(f"Valid number of records for PF: {valid_num_records_PF}")
    print(f"Not valid number of records for PF: {not_valid_num_records_PF}")
    print(f"Valid number of records for SC: {valid_num_records_SC}")
    print(f"Not valid number of records for SC: {not_valid_num_records_SC}")
    print(f"Average PF Score: {avg_PF_score * 10:.2f}")
    print(f"Average SC Score: {avg_SC_score * 10:.2f}")
    print(f"==========================")
