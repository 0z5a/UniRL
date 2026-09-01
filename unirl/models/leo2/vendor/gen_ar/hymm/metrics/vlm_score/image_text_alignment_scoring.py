import numpy as np
import os
import uuid
import hmac
import base64
from datetime import datetime, timezone
import hashlib
import json
import requests
from multiprocessing import Pool, Manager
from itertools import product
from collections import defaultdict
from tqdm import tqdm
import time
from hymm.metrics.vlm_score.utils import flatten_structured_info_into_list, convert_to_structured_matching_from_flattend_matching
from processors.image_kits import url_to_base64_and_mime


API_VERSION = "v2.03"
HOST = "trpc-gpt-eval.production.polaris"  # 生产

system_prompt = """
您是一个图文一致性核查专家, 请分析文字所描述的事物或现象是否在图片中存在, 如果是则返回英文Yes，否则返回No, 不需要赘述原因，直接返回Yes或No就行，也不要加个句号啥的，简洁就好
"""

def time2str(t):
    """将时间戳转换为字符串"""
    return datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')


def get_simple_auth(source, SecretId, SecretKey):
    date_time = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = "hmac id=\"" + SecretId + "\", algorithm=\"hmac-sha1\", headers=\"date source\", signature=\""
    sign_str = "date: " + date_time + "\n" + "source: " + source
    sign = hmac.new(SecretKey.encode(), sign_str.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    sign = auth + sign + "\""
    return sign, date_time


class Api:
    def __init__(self, hosts, user, apikey, model_marker, api_name):
        self.hosts = hosts
        self.user = user
        self.apikey = apikey
        self.model_marker = model_marker
        self.timeout = 305  # 超时时间
        self.apis = dict(
            data_eval=self.call_data_eval,
            chat_completions=self.call_chat_completions
        )
        self.api_name = api_name
        self.call_api = self.apis[api_name]
        self.hosts_calls = [0 for _ in range(len(hosts))]

    def get_header(self):
        source = 'xxxxxx'  # 签名水印值，可填写任意值
        # 如果模型是qwen，且user和apikey为None，则设置为qwen和qwen3vl; 防止没有传入user和apikey导致签名失败
        if "qwen" in self.model_marker.lower() and (self.user is None or self.apikey is None):
            self.user = "qwen"
            self.apikey = "qwen3vl"
        sign, dateTime = get_simple_auth(source, self.user, self.apikey)
        headers = {'Apiversion': API_VERSION, 'Authorization': sign, 'Date': dateTime, 'Source': source}
        return headers

    def call_data_eval(self, host, request_id, query_url, prompt):
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
                            "value": system_prompt
                        },
                        {
                            "type": "image_url",
                            "value": query_url
                        },
                        {
                            "type": "text",
                            "value": prompt
                        }
                    ]
                }
            ],
            "params": {
            },
            "timeout": 300
        }

        headers = dict(self.get_header())
        rsp = requests.post(url=base_url, headers=headers, json=data, timeout=self.timeout)
        return rsp, self.api_name

    def call_chat_completions(self, host, request_id, query_url, prompt):
        base_url = host + '/v1/chat/completions'
        for _ in range(10):
            base64_image, mime = url_to_base64_and_mime(query_url)
            if mime is not None:
                break
            time.sleep(0.5)
        else:
            raise ValueError(f"Failed to download image from {query_url}")

        data = {
            "model": self.model_marker,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime};base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
        }

        headers = dict(self.get_header())
        rsp = requests.post(url=base_url, headers=headers, json=data, timeout=self.timeout)
        return rsp, self.api_name

    def load_balance_host(self):
        min_calls = min(self.hosts_calls)
        idx = self.hosts_calls.index(min_calls)
        return idx


def get_result(image_url, prompt, api):
    max_retries = 50
    for i in range(max_retries):
        request_id = str(uuid.uuid4())
        ret = None
        try:
            host_idx = api.load_balance_host()
            ret, api_name = api.call_api(api.hosts[host_idx], request_id, image_url, prompt)
            ret = ret.json()
            if api_name == "data_eval":
                ans = ret["answer"][0]["value"]
            elif api_name == "chat_completions":
                ans = ret["choices"][0]["message"]["content"]
                if "</think>" in ans:
                    ans = ans.split("</think>")[-1].strip("\n ")
                request_id = ret["id"]
            else:
                raise ValueError(f"Unknown api_name: {api_name}")
            if ans:
                api.hosts_calls[host_idx] += 1
                return ans, request_id
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"[{time2str(time.time())}] [{request_id=}] [Try {i + 1}] [{ret=}] {e.__class__.__name__}: {e}")
            # raise e
    return None, None


def run_model(task, api):
    ans, request_id = get_result(task['img_url'], task['pt'], api)
    if ans is None:
        ans = '#INVALID', "None"
    return ans, request_id


def task_controller(input_list, k_per_sec=10):
    """带精确时间窗口的任务生成器"""
    interval = 1.0 / k_per_sec
    for item in input_list:
        yield item
        time.sleep(interval)  # 关键速率控制点[2,5](@ref)


def worker(task, api):
    """带索引记忆的工作进程"""
    try:
        result, request_id = run_model(task, api)
        return task['prompt_idx'], task['img_idx'], task['pt_idx'], result, request_id
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
        # raise e
        return task['prompt_idx'], task['img_idx'], task['pt_idx'], '#INVALID', "None"


def main_processor(input_list, api, max_workers=None, k_per_sec=10):
    """
    主处理流程
    :param max_workers: 进程数，默认CPU核心数[7](@ref)
    :param k_per_sec: 每秒最大调用量
    """
    max_workers = max_workers or os.cpu_count()
    
    # 创建共享字典用于结果存储
    with Manager() as manager:
        shared_dict = manager.dict()
        
        with Pool(processes=max_workers) as pool:
            # 创建速率控制的任务生成器
            task_gen = task_controller(input_list, k_per_sec)
            
            # 提交任务并集成进度条[2,6](@ref)
            futures = []
            with tqdm(total=len(input_list), desc="Processing by calling VLM", unit="task", position=1, leave=False) as pbar:
                for task in task_gen:
                    future = pool.apply_async(
                        worker, 
                        (task, api),
                        callback=lambda res: (
                            shared_dict.update({(res[0], res[1], res[2]): (res[3], res[4])}),
                            pbar.update(1)  # 实时进度更新[5,7](@ref)
                        )
                    )
                    futures.append(future)
                
                # 等待所有任务完成
                for future in futures:
                    future.get()
        
        # 重组嵌套数据结构[1,4](@ref)
        nested = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(str)
            )
        )
        nested = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(str)
            )
        )
        nested_id = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(str)
            )
        )
        for (grp, img, pt), (res, request_id) in shared_dict.items():
            nested[grp][img][pt] = res
            nested_id[grp][img][pt] = request_id

        # 转换为普通字典
        return defaultdict_to_regular(nested), defaultdict_to_regular(nested_id)

def defaultdict_to_regular(d):
    """递归转换defaultdict为普通dict"""
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_regular(v) for k, v in d.items()}
    return d


def numpy_masked_score(a, b):
    arr = np.array(a)
    mask = np.array(b, dtype=bool)
    masked = arr[:, mask]  # 按列过滤
    return (masked.sum(axis=1) / mask.sum()).tolist()


def find_lose_win_pairs(scores):
    '''
    Example:
        Input:  [0.2, 0.2, 0.3, 0.5]
        Return: [{'lose': 0, 'win': 3}, {'lose': 1, 'win': 3}]
    '''

    # 获取所有最低分和最高分索引
    min_val = min(scores)
    max_val = max(scores)
    min_indices = [i for i, x in enumerate(scores) if x == min_val]
    max_indices = [i for i, x in enumerate(scores) if x == max_val]
    
    # 生成所有可能的配对组合
    return [
        {'lose': lose, 'win': win} 
        for lose, win in product(min_indices, max_indices)
        if lose != win  # 排除同索引情况（根据需求可选）
    ]

def assert_no_nan(target_list):
    # 转换为numpy数组以支持向量化操作
    arr = np.array(target_list, dtype=np.float64)
    # 断言所有元素都不是NaN
    assert not np.any(np.isnan(arr)), "列表中存在NaN值"


def handle(json_list, api, num_workers, max_api_qps):
    if len(json_list) == 0:
        return []
    # Keys: ['prompt', 'prompt_idx', 'img_info_list', 'structured_semantic_points']
    # record_id, img_id, prompt_id 三重定位
    input_list = []

    # print('Preprocess the inputs (e.g., flatten and record flatten2structured cues) for fast concurrent api calling ...')
    for grp_idx, info_dict in enumerate(json_list):
        prompt_idx = info_dict['prompt_idx']
        img_urls = [item['url_cos'] for item in info_dict['img_info_list']]
        # 根据一级类目拍平prompt, 做好索引归属, 
        flattened_semantic_points, point_idx_to_taxonomy_key = flatten_structured_info_into_list(info_dict['structured_semantic_points'])
        # print(flattened_semantic_points)
        # 把拍平的语义点记录到json_list里面去, 下面可以复用, 同时记录结构化形式和平摊形式的对应关系，方便转化 (多进程并行用平摊形式，最后展现给客户用的结构化形式)
        json_list[grp_idx]['flattened_semantic_points'] = flattened_semantic_points
        json_list[grp_idx]['point_idx_to_taxonomy_key'] = point_idx_to_taxonomy_key
        for img_idx, img_url in enumerate(img_urls):
            for pt_idx, point in enumerate(flattened_semantic_points):
                input_list.append({'prompt_idx': prompt_idx, 'img_idx': img_idx, 'pt_idx': pt_idx, 'img_url': img_url, 'pt': point})
    # print('Done')

    # 并发请求API
    # print('start assessing image alignment ...')
    result, result_id = main_processor(
        input_list,
        api,
        max_workers=num_workers,
        k_per_sec=max_api_qps,
    )
    
    # 遍历组
    # 索引定位上 外层（grp的)idx没有意义, 因为目前读取group间记录的顺序是随机的, 用prompt_idx替代, 内层的img_idx和pt_idx有定位意义，本身就是在group 记录的数据结构内的idx
    # print(f'There are total {len(json_list)} info dict to process in json_list')
    for idx, info_dict in enumerate(json_list):
        prompt_idx = info_dict['prompt_idx']
        img_urls = [item['url_cos'] for item in info_dict['img_info_list']]
        candidate_semantic_points = info_dict['flattened_semantic_points']
        nr_imgs = len(img_urls)
        nr_pts = len(candidate_semantic_points)
        # Initialization
        json_list[idx]['semantic_points_matching'] = [[np.nan for _ in range(nr_pts)] for _ in range(nr_imgs)] # (nr_imgs, nr_pts)
        json_list[idx]['semantic_points_matching_request_id'] = [['None' for _ in range(nr_pts)] for _ in range(nr_imgs)] # (nr_imgs, nr_pts)
        json_list[idx]['semantic_points_matching_details'] = [['Miss' for _ in range(nr_pts)] for _ in range(nr_imgs)] # (nr_imgs, nr_pts)
        json_list[idx]['masked_equally_weighted_score'] = [0.0 for _ in range(nr_imgs)] # (nr_imgs,)
        json_list[idx]['score_certainty'] = 0.0 # scalar
        # record image-point valid api ret situation
        json_list[idx]['legal_mask'] = [[True for _ in range(nr_pts)] for _ in range(nr_imgs)] # (nr_imgs, nr_pts)

        # Fetch results from the api calling previously
        for img_idx in range(nr_imgs):
            # NOTE: 如果当前图没有任何返回语义点的推理记录, 那它完全没法和同组其他图的图相比较, 后续根据把图从win-lose选择中剔除
            if prompt_idx not in result:
                result[prompt_idx] = dict()
                result_id[prompt_idx] = dict()

            if img_idx not in result[prompt_idx]:
                print(f'|__group {prompt_idx}: image({img_idx}) lacks all semantic points alignment api results')
                json_list[idx]['legal_mask'][img_idx] = [False for _ in range(nr_pts)]
                json_list[idx]['semantic_points_matching_details'][img_idx] = ['Miss' for _ in range(nr_pts)]
                json_list[idx]['semantic_points_matching'][img_idx] = [np.nan for _ in range(nr_pts)]
                continue
            try:
                for pt_idx, point in enumerate(candidate_semantic_points):
                    # NOTE: 如果语义点询问丢包, 那就按点记录invalid
                    if (pt_idx not in result[prompt_idx][img_idx]) or (result[prompt_idx][img_idx][pt_idx] == '#INVALID'):
                        print(f'|__group {prompt_idx}: image({img_idx})-point({pt_idx}) api result misses')
                        json_list[idx]['semantic_points_matching_details'][img_idx][pt_idx] = 'Miss'
                        json_list[idx]['semantic_points_matching_request_id'][img_idx][pt_idx] = 'None'
                        json_list[idx]['semantic_points_matching'][img_idx][pt_idx] = np.nan
                        json_list[idx]['legal_mask'][img_idx][pt_idx] = False
                    else:
                        raw_ans = result[prompt_idx][img_idx][pt_idx]
                        request_id = result_id[prompt_idx][img_idx][pt_idx]
                        json_list[idx]['semantic_points_matching_request_id'][img_idx][pt_idx] = request_id
                        if 'yes' in raw_ans.lower():
                            int_ans = 1
                            json_list[idx]['semantic_points_matching_details'][img_idx][pt_idx] = raw_ans
                            json_list[idx]['semantic_points_matching'][img_idx][pt_idx] = int_ans
                        elif 'no' in raw_ans.lower():
                            int_ans = 0
                            json_list[idx]['semantic_points_matching_details'][img_idx][pt_idx] = raw_ans
                            json_list[idx]['semantic_points_matching'][img_idx][pt_idx] = int_ans
                        else:
                            print(f'|__group {prompt_idx}: image({img_idx})-point({pt_idx}) api result misses')
                            json_list[idx]['semantic_points_matching_details'][img_idx][pt_idx] = 'Miss'
                            json_list[idx]['semantic_points_matching'][img_idx][pt_idx] = np.nan
                            json_list[idx]['legal_mask'][img_idx][pt_idx] = False
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(f'- -bug happens {e}')
        # alignment score 计算, 考虑api丢包控制: 
        # 根据所有图对point的valid mask 取交再取计算分数(将来可视化时仅高亮参与评分的那些点 (不参与评分的点直接划删除线, 且不在计算score的分母内)
        legal_matrix = json_list[idx]['legal_mask']
        legal_pt_mask = [all(column) for column in zip(*legal_matrix)] # (nr_pts,) bool

        assert len(legal_pt_mask) == nr_pts

        score_certainty = sum(legal_pt_mask)/nr_pts if nr_pts > 0 else 0.0
        json_list[idx]['score_certainty'] = score_certainty 
        assert len(legal_pt_mask) == len(info_dict['flattened_semantic_points'])

        json_list[idx]['exact_scoring_points_mask'] = {point: legalty for legalty, point in zip(legal_pt_mask, info_dict['flattened_semantic_points'])}
        if score_certainty == 0.0:
            json_list[idx]['masked_equally_weighted_score'] = [np.nan for _ in range(nr_imgs)]
        else:
            json_list[idx]['masked_equally_weighted_score'] = numpy_masked_score(json_list[idx]['semantic_points_matching'], legal_pt_mask)

        # NOTE: json_list[idx]['semantic_points_matching'] 是当前组的所有N张图对平摊形式的K个prompts的alignment结果，需要得到结构化的版本方便客户在外面用自定义的metric自行加权这个去算vlm score
        # 只需要得到长为N的list of dict, 每个dict记录对应某张图片对K个prompt的结构化形式的alignment情况
        json_list[idx]['structured_semantic_points_matching'] = \
            convert_to_structured_matching_from_flattend_matching(
            json_list[idx]['semantic_points_matching'], 
            json_list[idx]['point_idx_to_taxonomy_key'],
            json_list[idx]['structured_semantic_points'],
            json_list[idx]['flattened_semantic_points'])

    # eliminate unused intermediate results
    keys_to_drop = [
        'point_idx_to_taxonomy_key',
        'parsed_semantic_points_with_filtering',
        'legal_mask',
        'flattened_semantic_points',
        'semantic_points_matching',
    ]
    for idx in range(len(json_list)):
        for k in keys_to_drop:
            if k in json_list[idx]:
                _ = json_list[idx].pop(k)

    # print('|__Done')
    return json_list


def image_text_alignment_scoring(json_list,
                                 num_workers=64,
                                 max_api_qps=100,
                                 model_marker="api_google_gemini-2.5-pro-preview-03-25",
                                 account="None",
                                 passwd="1234",
                                 host=HOST,
                                 api_name="data_eval"):
    if isinstance(host, str):
        hosts = ["http://{}:8080".format(host)]
    elif isinstance(host, list):
        hosts = ["http://{}:8080".format(h) for h in host]
    else:
        raise ValueError(f"host should be str or list, but got {type(host)}")
    api = Api(hosts, account, passwd, model_marker, api_name=api_name) # 生产
    output_json_list = handle(json_list, api, num_workers, max_api_qps)
    return output_json_list
