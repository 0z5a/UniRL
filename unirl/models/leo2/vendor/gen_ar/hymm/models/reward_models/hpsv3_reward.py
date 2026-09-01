import base64
import json
import requests
from io import BytesIO
from typing import Union, List

from PIL import Image


def image2b64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def pil_img2b64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


class HPSV3RewardModel(object):
    def __init__(self, url:str, port:str = "8080"):
        self.url = url
        self.port = port

    def request_hpsv3(self, image: Union[Image.Image, List[Image.Image]], prompt: Union[str, List[str]]):
        if isinstance(image, Image.Image):
            img_base64 = [pil_img2b64(image)]
        elif isinstance(image, List):
            img_base64 = [pil_img2b64(ele) for ele in image]
        
        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) != len(img_base64):
            raise ValueError(f"Prompt and image length mismatch: {len(prompt)} != {len(img_base64)}")
        
        data = [
            {"img_base64": img_base64, "prompt": prompt, "data_type": "IMAGE"} 
            for img_base64, prompt in zip(img_base64, prompt)
        ]
        response = requests.post(
            f"http://{self.url}:{self.port}/",
            json={
                "data_size": len(data),
                "data": data,
                "stats": ["hpsv3_server"],
                "stat_config": {"hpsv3_server": {}},
                "request_id": "request_test",
                "request_ts": 0,
                "source": "request_test",
                "bid": "BID_request_test",
                "principal": "jacksymao",
            },
            headers={"Content-Type": "application/json"},
        )

        # Parse the response
        # {'code': 200, 'message': 'success', 'request_id': 'request_test', 'data': [{'hpsv3': {'code': 200, 'value': 6.791212558746338, 'version': '20250807'}}]}
        scores = [0] * len(data)
        successes = [False] * len(data)
        try:
            for i, item in enumerate(response.json()["data"]):
                hpsv3_value = item["hpsv3"]["value"]
                scores[i] = hpsv3_value
                successes[i] = item["hpsv3"]["code"] == 200 and isinstance(hpsv3_value, float)
        except Exception as e:
            print(f"Error: {e}")
            return scores, successes
        return scores, successes
    
    def __call__(
            self,
            images: Union[Image.Image, List[Image.Image]],
            texts: Union[str, List[str]],
    ):
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]
        
        scores, successes = self.request_hpsv3(images, texts)
        return scores, successes


if __name__ == "__main__":
    reward_model = HPSV3RewardModel(url="xxx", port="xxx")
    image = Image.open("hymm/models/reward_models/my_imgs/src_将鹅卵石街道换成深灰色柏油路.png")
    prompt = "将鹅卵石街道换成深灰色柏油路"
    scores, successes = reward_model(image, prompt)
    print(scores)
    print(successes)

"""
PYTHONPATH=./ python3 hymm/models/reward_models/hpsv3_reward.py
"""
