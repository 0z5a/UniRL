import json
import requests
import time
import base64
import glob
import natsort
import argparse
from datetime import datetime
from io import BytesIO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--image_path_list", type=str, default="")
    parser.add_argument("--output_path", type=str, default="results.json")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--show_progress", action="store_true", help="Display progress during processing")
    return parser.parse_args()

def image2b64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def pil_img2b64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def has_repeated_text(text):
    splits = text.split(" ")
    last_n = 3
    last = splits[-last_n:]
    prev = splits[:-last_n]

    last_str = " ".join(last)
    prev_str = " ".join(prev)
    # calculate how many times the last n tokens appear in the previous text
    repetition_n = prev_str.count(last_str)
    if repetition_n > 1:
        return repetition_n, True
    else:
        return 0, False

def safe_eval(s):
    x = s.split(',')
    for i in range(len(x)):
        # if "'" > 2, replace the internal ' with \\'
        if x[i].count("'") > 2:
            first = x[i].find("'")
            last = x[i].rfind("'")
            if x[i][first + 1:last].count("\\'") == 0:
                x[i] = x[i][:first + 1] + x[i][first + 1:last].replace("'", "\\'") + x[i][last:]

    s = ','.join(x)
    try:
        result = eval(s)
        return result
    except Exception as e:
        print(f"Error evaluating string: {e}")
        return []


def request_one(url, sample):
    start_time = time.time()
    try:
        response_json = {}
        img_pil = sample["image"]
        b64 = pil_img2b64(img_pil)

        data = {
            "data_size": 1,
            "data": [
                {
                    "base64": b64,
                    "data_type": "IMAGE",
                    # "text": "Describe this image and output the results in json format. The results need to include: short_caption, medium_caption, long_caption, background, shot_type, style, light, atmosphere, composition, IP (Precisely quote clear and prominent text in the image; For illegible or obscured text, simply state 'text is present' at the location.Describe this image and output the results in json format. The results need to include: short_caption, medium_caption, long_caption, background, shot_type, style, light, atmosphere, composition, IP (Precisely quote clear and prominent text in the image; For illegible or obscured text, simply state 'text is present' at the location."
                    # "text": "Output the coordinates of the text in the image without any prefix",
                    # "text": "Recognize the text in the image, and directly output the recognition result without any prefix.",
                    "text": "Recognize all the text in the image, and directly output the recognition result without any prefix. Your output format should be a list of paragraph text: [text1, text2, text3, ...]. if there is no text in the image, just output an empty list. Do not output the description of the image.",
                    # "text": "检查图片中是否存在无意义的符号、乱码或无法辨认的文字（鬼画符）。请直接回答'是'或'否'，不要添加任何前缀或解释。如果图片中只有正常可读的文字，回答'否'；如果存在任何无法辨认的符号或乱码，回答'是'。",
                }
            ],
            "stats": [
                "mllm"
            ],
            "stat_config": {
                "mllm": {
                    "extra_body": {
                        "top_k": 1,
                        "repetition_penalty": 1.0
                    },
                    "max_tokens": 1024,
                    "top_p": 0.001,
                }
            },
            "request_id": "ocr_detection",
            "request_ts": time.time(),
            "source": "ocr_detection",
            "bid": "BID_ocr_detection",
            "principal": "josonchen"
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=data, headers=headers)
        response_json = response.json()
        sample["response"] = response_json
        pred_ocr = response_json['data'][0]['mllm']["value"]['response']
        if not pred_ocr.endswith("]"):
            repeat_n, has_repeat = has_repeated_text(pred_ocr)
            if has_repeat:
                print(f"processing sample pred_ocr has repeated text: {pred_ocr}")
                pred_ocr = pred_ocr[:pred_ocr.rfind(",")] + "]"
                ocr_list = safe_eval(pred_ocr)
                if len(ocr_list) == 1:
                    raise ValueError("pred_ocr is invalid")
                assert isinstance(ocr_list, list)
                # Remove duplicates while preserving order
                seen = set()
                unique_ocr = []
                for item in ocr_list:
                    if item not in seen:
                        seen.add(item)
                        unique_ocr.append(item)
                    # Update pred_ocr with deduplicated list
                pred_ocr = str(unique_ocr)
                print(f"deduplicated pred_ocr: {pred_ocr}")
            else:
                raise ValueError("pred_ocr is invalid")

        sample["pred_ocr"] = pred_ocr
        ocr_rlt = safe_eval(pred_ocr)
        sample["pred_ocr_json"] = json.dumps(ocr_rlt)
    except Exception as e:
        print(f"Error processing sample: {str(e)} for {response_json}")
        sample["error"] = str(object=e)
        sample["pred_ocr"] = str([]) if "pred_ocr" not in sample else sample["pred_ocr"]
        sample["pred_ocr_json"] = json.dumps([]) if "pred_ocr_json" not in sample else sample["pred_ocr_json"]

    end_time = time.time()
    sample["processing_time"] = end_time - start_time
    return sample


def request_batch(url, samples, max_workers=8, show_progress=False):
    """
    Process a batch of samples by making concurrent requests to the specified URL.

    Args:
        url (str): The URL to send requests to.
        samples (PIL Image): A list of dict containing pil images.
        max_workers (int, optional): Maximum number of concurrent requests. Defaults to 8.
        show_progress (bool, optional): Whether to display a progress bar. Defaults to False.

    Returns:
        list: The processed samples with caption information added.
    """
    import concurrent.futures

    # Import tqdm only if needed to avoid unnecessary dependency
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            print("Warning: tqdm not installed. Progress bar disabled.")
            show_progress = False

    # Define a worker function for each request
    def process_single_sample(sample):
        return request_one(url, sample)

    # Create futures with their index to maintain order
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, sample in enumerate(samples):
            futures.append(executor.submit(process_single_sample, sample))

        # Collect results in the original order
        results = []
        if show_progress:
            for future in tqdm(futures, total=len(samples), desc="Processing samples"):
                results.append(future.result())
        else:
            for future in futures:
                results.append(future.result())

    return results