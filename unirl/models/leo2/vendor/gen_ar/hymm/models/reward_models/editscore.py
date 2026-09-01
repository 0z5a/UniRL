import json
import pickle
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Union

import loguru
from PIL import Image

from hymm.models.reward_models.google_gemini_request import GoogleGeminiRewardModel


class RewardClient:
    """
    Pure Reward Client - Only responsible for communicating with proxy server
    """
    
    def __init__(self, proxy_host: str = "127.0.0.1", proxy_port: int = 23456, 
                 timeout: int = 300, max_retries: int = 3, logger=None):
        """
        Initialize client
        
        Args:
            proxy_host: Proxy server host address
            proxy_port: Proxy server port
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.proxy_url = f"http://{proxy_host}:{proxy_port}"
        self.timeout = timeout
        self.max_retries = max_retries
        if logger is None:
            self.logger = loguru.logger
        else:
            self.logger = logger
        self.logger.info(f"Initialize Reward client: {self.proxy_url}")
    
    def evaluate(self, input_images, output_image, meta_datas, server_type: str = 'geneval'):
        """
        Evaluate images and return rewards
        
        Args:
            input_images: List of input image byte data
            output_image: List of output image byte data
            meta_datas: List of metadata
            server_type: Server type ('geneval', 'ocr', etc.)
            
        Returns:
            tuple: (scores, rewards, reasoning, meta_data) 
            - scores: List of scores
            - rewards: List of rewards
            - reasoning: List of reasoning results
            - meta_data: List of metadata
        """
        try: 
            if not output_image:
                return [], [], [], []
            
            # Prepare request data
            request_data = {
                'input_images': input_images,
                'output_image': output_image,
                'meta_datas': meta_datas,
                'server_type': server_type  
            }
            
            # Retry logic
            last_exception = None
            for attempt in range(self.max_retries):
                try:
                    # Serialize and send
                    pickled_data = pickle.dumps(request_data)
                    response = requests.post(
                        self.proxy_url,
                        data=pickled_data,
                        headers={'Content-Type': 'application/octet-stream'},
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        # Parse results
                        result = pickle.loads(response.content)
                        scores = result.get('scores', [])
                        rewards = result.get('rewards', [])
                        reasoning = result.get('reasoning', [])
                        meta_data = result.get('meta_data', [])
                        
                        # Basic validation
                        if len(scores) != len(output_image) or len(rewards) != len(output_image):
                            self.logger.warning(f"Return data length mismatch: expected {len(output_image)}, got scores={len(scores)}, rewards={len(rewards)}")
                        
                        return scores, rewards, reasoning, meta_data
                    else:
                        self.logger.error(f"HTTP error: {response.status_code}")
                        last_exception = RuntimeError(f"HTTP {response.status_code}")
                        
                except requests.exceptions.Timeout as e:
                    self.logger.error(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                    last_exception = e
                    
                except Exception as e:
                    self.logger.error(f"Request exception: {e} (attempt {attempt + 1}/{self.max_retries})")
                    last_exception = e
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            
            self.logger.error(f"All retries failed, last exception: {last_exception}")
            return None, None, None, None
        except Exception as e:
            self.logger.error(f"Evaluation exception: {e}")
            return None, None, None, None
    
    def ping(self) -> bool:
        """Check if server is reachable"""
        try:
            response = requests.get(f"{self.proxy_url}/ping", timeout=5)
            return response.status_code == 200
        except:
            return False

# Convenience function
def evaluate_images(input_images: List[bytes], output_image: List[bytes], meta_datas: List[Dict[str, Any]], 
                   proxy_host: str = "127.0.0.1", proxy_port: int = 23456,
                   server_type: str = 'vlm') -> Optional[Tuple[List[float], List[float], List[str], List[Dict]]]:
    """
    Convenience function: directly evaluate images
    """
    client = RewardClient(proxy_host, proxy_port, timeout=600, max_retries=1)
    return client.evaluate(input_images, output_image, meta_datas, server_type)


class EditScoreRewardModel(object):
    """
    EditScore Reward Model Wrapper
    """
    def __init__(
        self, 
        gg_app_id,
        gg_app_key,
        gg_api_version,
        gg_model_marker,
        gg_max_retries: int = 3,
        gg_timeout: int = 30,
        proxy_host: str = "127.0.0.1", 
        proxy_port: int = 23456, 
        timeout: int = 300, 
        max_retries: int = 3, 
        logger=None,
        resize=None,
        resize_mode: str = "square",  # square, shortest_side, longest_side
        translate_before_eval: bool = True,
    ):
        self.client = RewardClient(proxy_host, proxy_port, timeout, max_retries, logger)
        self.translator = GoogleGeminiRewardModel(
            app_id=gg_app_id,
            app_key=gg_app_key,
            api_version=gg_api_version,
            task_name="Translate_Ch_to_En",
            logger=logger,
            model_marker=gg_model_marker,
        )
        self.gg_max_retries = gg_max_retries
        self.gg_timeout = gg_timeout
        if logger is None:
            self.logger = loguru.logger
        else:
            self.logger = logger
        
        self.resize = resize
        self.translate_before_eval = translate_before_eval
        self.resize_mode = resize_mode

    def __call__(
        self,
        input_images: Union[Image.Image, List[Image.Image]],
        output_images: Union[Image.Image, List[Image.Image]],
        texts: Union[str, List[str]],
        rank: int = 0,
    ):
        try:
            if isinstance(input_images, Image.Image):
                input_images = [[input_images]]
            if isinstance(input_images, list) and isinstance(input_images[0], Image.Image):
                input_images = [[img] for img in input_images]
            if isinstance(output_images, Image.Image):
                output_images = [output_images]
            if isinstance(texts, str):
                texts = [texts]
            if self.resize is not None:
                if self.resize_mode == "shortest_side":
                    input_images = [
                        [img.resize((self.resize, int(img.height * self.resize / img.width))) if img.width < img.height 
                         else img.resize((int(img.width * self.resize / img.height), self.resize)) for img in img_list] 
                        for img_list in input_images
                    ]
                    output_images = [
                        img.resize((self.resize, int(img.height * self.resize / img.width))) if img.width < img.height 
                        else img.resize((int(img.width * self.resize / img.height), self.resize)) for img in output_images
                    ]
                elif self.resize_mode == "longest_side":
                    input_images = [
                        [img.resize((self.resize, int(img.height * self.resize / img.width))) if img.width > img.height 
                         else img.resize((int(img.width * self.resize / img.height), self.resize)) for img in img_list] 
                        for img_list in input_images
                    ]
                    output_images = [
                        img.resize((self.resize, int(img.height * self.resize / img.width))) if img.width > img.height 
                        else img.resize((int(img.width * self.resize / img.height), self.resize)) for img in output_images
                    ]
                elif self.resize_mode == "square":
                    input_images = [
                        [img.resize((self.resize, self.resize)) for img in img_list] for img_list in input_images
                    ]
                    output_images = [
                        img.resize((self.resize, self.resize)) for img in output_images
                    ]
                else:
                    raise ValueError(f"Unknown resize_mode: {self.resize_mode}")

            successes = [True] * len(texts)
            if self.translate_before_eval:
                # Translate texts to English, keep order
                with ThreadPoolExecutor(max_workers=len(texts)) as executor:
                    future_to_index = {
                        executor.submit(
                            self.translator.translate_ch_en, 
                            text, 
                            try_times=self.gg_max_retries, 
                            timeout=self.gg_timeout
                        ): idx for idx, text in enumerate(texts)
                    }
                    translated_texts = [None] * len(texts)
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            translated_text = future.result()
                            translated_texts[idx] = translated_text
                        except Exception as e:
                            self.client.logger.error(f"Translation failed for text index {idx}: {e}")
                            translated_texts[idx] = texts[idx]  # Fallback to original text
                for t_t in translated_texts:
                    if t_t is None or t_t == "#INVALID#":
                        successes[idx] = False
            else:
                translated_texts = texts
                successes = [True] * len(texts)

            # Prepare meta data
            meta_datas = [
                json.dumps({"index": i, "rank": rank, "instruction": translated_texts[i]}) for i in range(len(translated_texts))
            ]
            # self.logger.info(f"Meta datas for evaluation: {meta_datas}")
            
            # Evaluate
            result = self.client.evaluate(
                input_images=input_images,
                output_image=output_images,
                meta_datas=meta_datas,
                server_type='vlm',
            )
            # print(f"111result: {result}")
            if result == None:
                return [0.0]*len(successes), [False]*len(successes)
            else:
                scores, rewards, reasoning, meta_data = result
                self.logger.info(f"EditScore rewards: {rewards}")
                self.logger.info(f"EditScore reasoning: {reasoning}")
                self.logger.info(f"EditScore meta_data: {meta_data}")
                return rewards, successes
        except Exception as e:
            self.logger.error(f"EditScore evaluation exception: {e}")
            return [0.0]*len(texts), [False]*len(texts)
