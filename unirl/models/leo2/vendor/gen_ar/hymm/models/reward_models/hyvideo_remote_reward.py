from typing import Any, Dict, List, Optional, Union
import argparse
import json
import time

import loguru
import requests


logger = loguru.logger

# deploy_vllm_server_new.sh puts a single **dispatcher** (reward_dispatcher.py) in
# front of every replica on ONE address (<chief_ip>:8100 by default). The dispatcher
# fans a batched request out to the healthy replicas, retries failures and re-probes
# dead ones, so a caller only needs the single dispatcher URL and no longer does
# client-side load balancing (the old random/min per-client picking is gone).
DEFAULT_SERVER_URL = "30.203.134.182"
DEFAULT_DISPATCHER_PORT = 8103

Logit2Dim = {
    0: "TA",
    1: "VQ",
    2: "MQ",
    3: "ID",
    4: "AVC"
}

# Logit2Dim = {
#     0: "SP",
#     1: "PHY",
#     2: "DQ",
#     3: "TA",
#     4: "AVC"
# }


class HyVideoRewardRemote:
    """Thin client for the single-entry reward dispatcher.

    Load balancing is now done server-side by the dispatcher, so this client
    simply points at one URL and sends the whole (videos, prompts) batch in a
    single ``/v1/reward/logits`` request.
    """

    def __init__(self,
        server_url: str,
        mode: str = "random",       # kept for backward-compat; dispatcher does LB now (unused)
        gpus_per_node: int = 8,     # kept for backward-compat (unused)
        tp_size: int = 1,           # kept for backward-compat (unused)
        max_retries: int = 10,
        port: int = 8100,
        timeout: float = 3600.0,
        model_name: str = "qwen3-vl-bt",
        ) -> None:
        """
        Args:
            server_url: str, dispatcher host (optionally "host:port"). Legacy list-like
                strings / trailing ":8" are tolerated (only the first host is used).
            mode: str, kept for backward-compat; ignored (dispatcher balances load).
            gpus_per_node/tp_size: kept for backward-compat; ignored.
            max_retries: int, maximum number of client-side retries for failed requests.
            port: int, dispatcher port (default 8100).
            timeout: float, per-request timeout in seconds.
            model_name: str, model name forwarded in the payload.
        """
        host, parsed_port = self._parse_host_port(server_url, port)
        self.base_url = f"http://{host}:{parsed_port}"
        self.max_retries = max_retries
        self.timeout = timeout
        self.model_name = model_name
        # dispatcher LB replaces client-side picking; kept only for logging / compat.
        self.mode = mode

        # Cluster IPs must NOT go through the company proxy.
        self.session = requests.Session()
        self.session.trust_env = False

        logger.info(f"Using reward dispatcher at {self.base_url} (server-side load balancing)")
        self._log_health()

    @staticmethod
    def _parse_host_port(server_url: str, default_port: int):
        """Extract (host, port) from a possibly-sloppy server_url string.

        Accepts a single host, "host:port", legacy trailing ":8", or a list-like
        string of several IPs (only the first is used, since the dispatcher is a
        single entry point).
        """
        raw = (server_url or "").strip()
        for sep in ("\n", "\t", ";", " ", ","):
            raw = raw.replace(sep, ",")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        first = parts[0] if parts else "127.0.0.1"
        # strip a legacy trailing ":8" convention.
        if first.endswith(":8"):
            first = first[:-2]
        if ":" in first:
            host, _, port_str = first.rpartition(":")
            try:
                return host, int(port_str)
            except ValueError:
                return first.replace(":8", ""), default_port
        return first, default_port

    def _log_health(self):
        """Best-effort health probe; never fatal (dispatcher may still be warming up)."""
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=30)
            r.raise_for_status()
            h = r.json()
            logger.info(
                f"Dispatcher health: num_healthy={h.get('num_healthy')} status={h.get('status')}"
            )
            if h.get("num_healthy", 0) == 0:
                logger.warning("Dispatcher reports 0 healthy replicas.")
        except Exception as e:
            logger.warning(f"Failed to query dispatcher /health at {self.base_url}: {e}")

    def get_logits_from_vllm(self, messages: list, model_name: str = None,
                             task_label: Optional[List[int]] = None) -> Union[
                                 List[float], List[List[float]], Dict[str, float], List[Dict[str, float]]]:
        """
        Get logits (reward scores) from the dispatcher via /v1/reward/logits.

        Args:
            messages: List of messages in OpenAI format (one per sample).
            model_name: Model name (defaults to self.model_name).
            task_label: Optional task_label list forwarded to the model.

        Returns:
            The raw ``logits`` field from the response. Its shape depends on the
            server's ``return_dict`` setting and the batch size:
              * return_dict=False: single -> [float], batch -> [[float], ...]
              * return_dict=True : single -> {dim: score}, batch -> [{...}, ...]
            ``reward()`` normalizes all of these into a list of {dim: score} dicts.
        """
        model_name = model_name or self.model_name
        url = f"{self.base_url}/v1/reward/logits"
        payload: Dict[str, Any] = {"model": model_name, "messages": messages}
        if task_label is not None:
            payload["task_label"] = task_label

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                if attempt != 0:
                    logger.info(f"Retrying request (attempt {attempt + 1}/{self.max_retries})...")
                response = self.session.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()

                logits = response.json()["logits"]

                if attempt != 0:
                    logger.info(f"Request succeeded (attempt {attempt + 1})")

                # Return the raw payload; reward() handles the vector/dict shapes.
                return logits

            except Exception as e:
                last_exc = e
                logger.error(f"Dispatcher request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = attempt + 1
                    logger.info(f"  等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        raise last_exc

    def _parse_score(self, vec):
        if not isinstance(vec, (list, tuple)):
            return None

        reward_dict = {}
        for idx, score in enumerate(vec):
            if idx in Logit2Dim:
                reward_dict[Logit2Dim[idx]] = score

        return reward_dict

    def _to_entry(self, item):
        """Coerce one per-sample result into a {dim_name: score} dict.

        Handles both server ``return_dict`` shapes: an already-named
        ``{dim: score}`` dict (passed through), or a raw score vector ``[float]``
        (mapped via ``Logit2Dim``).
        """
        if isinstance(item, dict):
            return item
        if isinstance(item, (list, tuple)):
            return self._parse_score(item)
        raise ValueError(f"Unsupported per-sample logits item: {type(item)}")

    def _normalize_logits(self, raw):
        """Normalize the raw ``logits`` payload into a list of per-sample dicts.

        Accepts the four shapes the reward service / dispatcher may emit:
        single vector ``[float]``, batch ``[[float], ...]``, single dict
        ``{dim: score}``, or batch ``[{...}, ...]``.
        """
        # single sample, return_dict=True
        if isinstance(raw, dict):
            return [raw]
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"Unsupported logits payload: {type(raw)}")
        if len(raw) == 0:
            return []
        first = raw[0]
        # batch: list of dicts or list of vectors
        if isinstance(first, (dict, list, tuple)):
            return [self._to_entry(x) for x in raw]
        # single sample, return_dict=False: a flat vector of scalars
        return [self._parse_score(raw)]

    def reward(self, videos, prompts, task_label: Optional[List[int]] = None):
        """
        Score a batch in ONE dispatcher request (the dispatcher does the fan-out).

        Args:
            videos: List[str], list of video paths
            prompts: List[str], the prompts
            task_label: Optional[List[int]], forwarded to the model if provided.

        Returns:
            One {dim_name: score} dict per input.
        """
        assert len(videos) == len(prompts), "videos and prompts must be the same length"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "prompt", "prompt": prompt},
                    {"type": "video", "video": video},
                ],
            }
            for video, prompt in zip(videos, prompts)
        ]

        raw = self.get_logits_from_vllm(messages, task_label=task_label)
        entries = self._normalize_logits(raw)
        if len(entries) != len(videos):
            logger.warning(f"reward service returned {len(entries)} results for {len(videos)} inputs")
        return entries



def hyreward_remote_reward(server_url, port, video_path, prompt, mode="random"):
    reward_model = HyVideoRewardRemote(server_url=server_url, port=port, mode=mode, gpus_per_node=8, tp_size=1)
    for i in range(3):
        try:
            reward = reward_model.reward(videos=[video_path], prompts=[prompt])[0]
            return server_url, reward
        except requests.exceptions.ReadTimeout as e:
            print(f"ReadTimeout for {server_url}: {e}")
            time.sleep(i)
    return server_url, {}


if __name__ == "__main__":
    '''
    DEFAULT_SERVER_URL = "29.185.219.22"
    reward_model = HyVideoRewardRemote(server_url=DEFAULT_SERVER_URL, mode="random", gpus_per_node=8, tp_size=1)
    video_path = "/apdcephfs_nj8/share_301739632/jodai/hunyuan_text2video_data_preprocess/video/files/26.mp4"
    prompt = "A nurse prepares a vaccine syringe with calm precision before administering it to a smiling child, the entire process captured in a hyper-realistic style."
    rewards = reward_model.reward(videos=[video_path], prompts=[prompt])
    print(rewards)
    '''
    args = argparse.ArgumentParser()
    args.add_argument("--url", default="29.162.225.121", type=str)
    args.add_argument("--port", default=8103, type=int)
    args.add_argument("--video_path", default="/apdcephfs_zwfy/share_303937731/dylanruili/project/HYVideoReWard/reward_model/demo/videos/e45574d7b0ec1b7b1c2941a38da85454.mp4", type=str)
    args.add_argument("--prompt", default="A nurse prepares a vaccine syringe with calm precision before administering it to a smiling child, the entire process captured in a hyper-realistic style.", type=str)
    args.add_argument("--output", type=str)
    args.add_argument("--mode", type=str, default="random", help="Kept for backward-compat; ignored (dispatcher balances load)")
    args = args.parse_args()
    result = hyreward_remote_reward(args.url, args.port, args.video_path, args.prompt, args.mode)
    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    else:
        print(result)
