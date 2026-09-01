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
import random
from pathlib import Path
from loguru import logger
from api.cos import p_upload_list
from tqdm import tqdm
from hymm.constants import ASSETS_BASE
from processors.image_kits import url_to_base64_and_mime, local_path_to_base64_and_mime

API_VERSION = "v2.03"


prompt_sc = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}
RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.


From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where \'score1\' evaluates the editing success and \'score2\' evaluates the degree of overediting.

Editing instruction: <instruction>"""

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
            'key': f'jarvizhang/metrics/vie_score/{today}_{model_name}/{file_path.name}'
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
    def __init__(self, hosts, user, apikey, model_marker, api_name):
        self.hosts = hosts
        self.user = user
        self.apikey = apikey
        self.model_marker = model_marker
        self.timeout = 3600  # 超时时间
        self.apis = dict(
            data_eval=self.call_data_eval,
            chat_completions=self.call_chat_completions
        )
        self.api_name = api_name
        self.call_api = self.apis[api_name]
        self.hosts_calls = [0 for _ in range(len(hosts))]

    def get_header(self):
        source = 'xxxxxxx'  # 签名水印值，可填写任意值
        sign, dateTime = get_simple_auth(source, self.user, self.apikey)
        headers = {'Apiversion': API_VERSION, 'Authorization': sign, 'Date': dateTime, 'Source': source}
        return headers

    def call_data_eval(self, host, request_id, prompt, img1_url, img2_url):
        base_url = host + '/api/v1/data_eval'
        data = {
            "request_id": request_id,
            "model_marker": self.model_marker,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "value": prompt
                        },
                        {
                            "type": "image_url",
                            "value": img1_url
                        },
                        {
                            "type": "image_url",
                            "value": img2_url
                        }
                    ]
                }
            ],
            "params": {},
            "timeout": 6000
        }
        headers = dict(self.get_header())
        rsp = requests.post(url=base_url, headers=headers, json=data, timeout=self.timeout)
        return rsp, self.api_name

    def call_chat_completions(self, host, request_id, prompt, img1_url, img2_url):
        base_url = host + '/v1/chat/completions'

        for _ in range(10):
            if Path(img1_url).exists():
                base64_image1, mime1 = local_path_to_base64_and_mime(img1_url)
            else:
                base64_image1, mime1 = url_to_base64_and_mime(img1_url)
            if mime1 is not None:
                break
            time.sleep(0.5)
        else:
            raise ValueError(f"Failed to download image from {img1_url}")

        for _ in range(10):
            if Path(img2_url).exists():
                base64_image2, mime2 = local_path_to_base64_and_mime(img2_url)
            else:
                base64_image2, mime2 = url_to_base64_and_mime(img2_url)
            if mime2 is not None:
                break
            time.sleep(0.5)
        else:
            raise ValueError(f"Failed to download image from {img1_url}")

        data = {
            "model": self.model_marker,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime1};base64,{base64_image1}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime2};base64,{base64_image2}"
                            }
                        },
                    ]
                }
            ],
        }

        headers = dict(self.get_header())
        rsp = requests.post(url=base_url, headers=headers, json=data, timeout=self.timeout)
        return rsp, self.api_name

    def load_balance_host(self):
        # min_calls = min(self.hosts_calls)
        # min_indices = [i for i, calls in enumerate(self.hosts_calls) if calls == min_calls]
        host = random.choice(self.hosts)
        return self.hosts.index(host)


def match_score(text):
    # 正则表达式匹配 "score" 后面的列表
    pattern = r'"score"\s*:\s*(\[[^\]]*\])'

    match = re.search(pattern, text)
    if match:
        score_str = match.group(1)

        # 如果需要转成Python列表
        import ast
        score_list = ast.literal_eval(score_str)
        # print(score_list)  # 输出: [4, 4]
        return score_list
    else:
        return None


def get_score(x):
    try:
        score_list = match_score(x)
        assert score_list[0] >= 0 and score_list[1] >= 0, "Score should be non-negative"
        return score_list[0], score_list[1]
    except Exception as e:
        print(f"Error processing text: {x}")
        print(e)
        return -1, -1


def log_score_io(row, host, request_id, img1, img2, instruction, sc_res, prefix="Score"):
    logger.info(
        f"[{prefix}][InputOutput][index={row.get('index', 'NA')}] "
        f"host={host}, request_id={request_id}\n"
        f"prompt={instruction}\n"
        f"src_img={img1}\n"
        f"tgt_img={img2}\n"
        f"response={sc_res}"
    )


def process_row(row, api: Api, queue):
    img1_url = row["src_img_path"]
    img2_url = row["tgt_img_path"]
    instruction = row['prompt']

    max_retry = 100
    sc_res = None
    request_id = None
    for i in range(max_retry):
        ans = None
        try:
            host_idx = api.load_balance_host()
            # logger.info(
            #     f"[BeforeCall][index={row.get('index', 'NA')}] "
            #     f"selected_host={api.hosts[host_idx]}, hosts_calls={api.hosts_calls}"
            # )
            request_id = str(uuid.uuid4())
            ret, api_name = api.call_api(
                api.hosts[host_idx], request_id, prompt_sc.replace('<instruction>', instruction), img1_url, img2_url
            )
            ret = ret.json()
            if api_name == "data_eval":
                sc_res = ret["answer"][0]["value"]
            elif api_name == "chat_completions":
                sc_res = ret["choices"][0]["message"]["content"]
                request_id = ret["id"]
            else:
                raise ValueError(f"Unknown api_name: {api_name}")
            log_score_io(row, api.hosts[host_idx], request_id, img1_url, img2_url, instruction, sc_res)
            api.hosts_calls[host_idx] += 1
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(ans if ans is not None else f"{e.__class__.__name__}: {e}\nInstruction: {instruction}")
            if i < max_retry - 1:
                time.sleep(0.5)

    if sc_res is not None:
        editing_score, consistency_score = get_score(sc_res)

        result = [
            row['index'], row['prompt'], row['src_img_path'], row['tgt_img_path'], editing_score, consistency_score,
            sc_res, request_id,
        ]
        queue.put(result)
    else:
        logger.error(f"Error processing text: {row}")


def run_smoke_test(api: Api, row, max_retry=3):
    img1_url = row["src_img_path"]
    img2_url = row["tgt_img_path"]
    instruction = row["prompt"]

    for i in range(max_retry):
        try:
            host_idx = api.load_balance_host()
            logger.info(
                f"[SmokeTest][BeforeCall] selected_host={api.hosts[host_idx]}, hosts_calls={api.hosts_calls}"
            )
            request_id = str(uuid.uuid4())
            ret, api_name = api.call_api(
                api.hosts[host_idx],
                request_id,
                prompt_sc.replace("<instruction>", instruction),
                img1_url,
                img2_url,
            )
            ret = ret.json()
            if api_name == "data_eval":
                sc_res = ret["answer"][0]["value"]
            elif api_name == "chat_completions":
                sc_res = ret["choices"][0]["message"]["content"]
                request_id = ret["id"]
            else:
                raise ValueError(f"Unknown api_name: {api_name}")
            log_score_io(
                row,
                api.hosts[host_idx],
                request_id,
                img1_url,
                img2_url,
                instruction,
                sc_res,
                prefix="SmokeTest",
            )
            if not isinstance(sc_res, str) or sc_res.strip() == "":
                raise ValueError(f"Smoke test got empty response: {ret}")
            api.hosts_calls[host_idx] += 1
            logger.info("Smoke test passed. Start concurrent scoring.")
            return
        except Exception as e:
            logger.warning(f"Smoke test failed ({i + 1}/{max_retry}): {e}")
            if i < max_retry - 1:
                time.sleep(0.5)

    raise SystemExit("Smoke test failed, exit before concurrent scoring.")


def writer_process(queue, output_file):
    with open(output_file, mode='a', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file)
        while True:
            result = queue.get()
            if result == "DONE":
                break
            writer.writerow(result)


def format_assets_base_path(img_path):
    if isinstance(img_path, str) and img_path.startswith('{ASSETS_BASE}'):
        return img_path.format(ASSETS_BASE=ASSETS_BASE)
    return img_path


def call_gemini_sc(args, df, output_csv):
    hosts = "trpc-gpt-eval.production.polaris" if args.hosts is None else args.hosts
    if isinstance(hosts, str):
        hosts = ["http://{}:8000".format(hosts)]
    else:
        hosts = ["http://{}:8000".format(h) for h in hosts]
    app_id = os.getenv("VLM_SCORE_APP_ID")
    token = os.getenv("VLM_SCORE_TOKEN")
    if app_id is None or token is None:
        raise ValueError("Please set the environment variables VLM_SCORE_APP_ID and VLM_SCORE_TOKEN.")
    
    # 可能还需要加新的 model_marker，先写成 list
    if args.model_marker in ['Qwen3-VL-235B-A22B-Thinking', "Qwen35-397B-A17B-FP8"]:
        assert args.hosts is not None and args.api_name == 'chat_completions'
    api = Api(hosts, app_id, token, args.model_marker, args.api_name)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        out_df = pd.read_csv(output_csv, header=0, index_col='index')
        done_indices = set(out_df.index)
    except Exception as e:
        print(e)
        print(f"It will be created to save the score results.")
        out_df = pd.DataFrame()
        done_indices = set()

    new_rows = df[~df.index.isin(done_indices)].to_dict('records')
    print(f"Total {len(new_rows)} new rows to process.")
    if len(new_rows) == 0:
        logger.info("No new rows to process.")
        return pd.read_csv(output_csv, header=0)

    smoke_test_n = max(1, args.smoke_test_n)
    smoke_rows = new_rows[:smoke_test_n]
    logger.info(f"Running {len(smoke_rows)} smoke test request(s) before concurrent scoring...")
    for i, smoke_row in enumerate(smoke_rows, start=1):
        logger.info(f"[SmokeTest] case {i}/{len(smoke_rows)}")
        run_smoke_test(api, smoke_row)
    if args.smoke_test_only:
        logger.info("Smoke test only mode enabled. Exit before concurrent scoring.")
        return None

    if out_df.empty:
        with open(output_csv, mode='w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'index', 'prompt', 'src_img_path', 'tgt_img_path', 'editing_score', 'consistency_score',
                'response', 'request_id',
            ])

    manager = Manager()
    queue = manager.Queue()

    writer_proc = Process(target=writer_process, args=(queue, output_csv))
    writer_proc.start()

    # 使用 functools.partial 传递额外的参数
    process_row_with_args = partial(process_row, api=api, queue=queue)

    # 使用多进程处理并显示进度条
    process_map(process_row_with_args, new_rows, max_workers=args.num_workers, chunksize=1)

    queue.put("DONE")
    writer_proc.join()

    return pd.read_csv(output_csv, header=0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True, help="Directory with AI generated images")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--testset", type=str, default="data/test/editing_arena_250718.csv", help="Testset csv file")
    parser.add_argument("--work-dir", type=str, default="__eval/vie_score/editing_arena_250718", help="Working directory")
    parser.add_argument("--only-calc-score", action='store_true', help="Only calculate score")
    parser.add_argument("--model-marker", type=str, default='api_google_gemini-2.5-pro',)
    parser.add_argument("--num-workers", type=int, default=32, help="Number of workers for local inference.")
    parser.add_argument("--api-name", type=str, default='data_eval', choices=["data_eval", "chat_completions"], help="Name of api to use.")
    parser.add_argument("--hosts", type=str, nargs='+', help="Custom host for the API.")
    parser.add_argument("--smoke-test-only", action='store_true', help="Run only the smoke test and exit.")
    parser.add_argument("--smoke-test-n", type=int, default=1, help="Number of smoke test requests before concurrent scoring.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # 创建工作目录
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_marker_repr = args.model_marker.replace("-", "_").replace(".", "_")
    output_file = work_dir / args.model_name / f"vie_score_{model_marker_repr}.csv"

    # 读取图像目录
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")
    image_files = list(image_dir.glob("*.png"))
    if len(image_files) == 0:
        raise ValueError(f"No PNG images found in the directory {image_dir}.")
    id2file = {int(f.stem.split('_')[0]): f for f in image_files}

    # 读取测试集并校验
    test_df = pd.read_csv(args.testset, header=0)
    test_df = test_df.apply(
        lambda col: col.map(format_assets_base_path) if col.dtype == "object" else col
    )
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

    test_df = test_df.dropna(subset=['src_img_path', 'tgt_img_path', 'prompt'])
    print(f"Total {len(test_df)} records after dropping NaN values.")
    if args.only_calc_score:
        score_df = pd.read_csv(output_file, header=0)
    else:
        score_df = call_gemini_sc(args, test_df, output_file)
        if args.smoke_test_only:
            logger.info("Smoke test only finished successfully.")
            raise SystemExit(0)

    avg_editing_score = score_df['editing_score'].mean()
    avg_consistency_score = score_df['consistency_score'].mean()
    valid_num_records = len(score_df[(score_df['editing_score'] >= 0) & (score_df['consistency_score'] >= 0)])
    print(f"==========================")
    print(f"Model name: {args.model_name}")
    print(f"Valid number of records: {valid_num_records}")
    print(f"Average Editing Score: {avg_editing_score * 10:.2f}")
    print(f"Average Consistency Score: {avg_consistency_score * 10:.2f}")
    print(f"==========================")
