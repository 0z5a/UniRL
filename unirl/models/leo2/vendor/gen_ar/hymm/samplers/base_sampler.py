import datetime
import inspect
import json
import math
import random
import sys
import time
from itertools import chain
from pathlib import Path
from typing import Dict, Any, List, Union
import os
import loguru
from easydict import EasyDict

import numpy as np
import pandas as pd
import csv
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from hymm.data_kits.csv_dataset import CSVDataset
from hymm.data_kits.datasets import load_dataset
from hymm.data_kits.datasets.label import LabelDataset
from hymm.data_kits.datasampler import DistributedSamplerFix
from hymm.utils.file_utils import safe_save_file
from hymm.utils.helpers import to_2tuple
from ..config import parse_args, add_ptm_args, sanity_check_args
from ..models import build_model
from ..utils.eval_utils import batch_data_repr
from ..utils.file_utils import safe_file, get_next_available_save_id, save_to_json
from ..utils.helpers import default, default_dtype, ensure_list
from ..utils.torch_utils import PRECISION_TO_TYPE, set_reproducibility
from ..utils.torch_utils import load_state_dict
from hymm.utils.env import should_use_external_model
from hymm.core.global_vars import get_nccl_timeout

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


def setup_distributed_initialize(args, mode, timeout=None):

    print(f"args: {args}, mode: {mode}")
    if timeout is None:
        timeout = get_nccl_timeout()
    if mode == 'pure_torch':
        dist.init_process_group("nccl", timeout=timeout)
        torch.cuda.set_device(dist.get_node_local_rank())
        from hymm.parallelism.parallel_states import init_parallel_state
        from hymm.parallelism.engines.gemini_parallel import GeminiParallelEngine
        parallel_dims = init_parallel_state(
            dp_replicate=-1, # world_size//8 在单个Node下进行切片
            dp_shard=-1,
            sp=1,
            tp=1,
            pp=args.pp_size,
            ep=args.ep_size,
            world_size=dist.get_world_size(),
        )
        parallel_dims.build_mesh('cuda')
        from accelerate.utils import set_seed
        dp_mesh = parallel_dims.dp_mesh
        seed = dp_mesh.get_local_rank()
        loguru.logger.info(f'{seed=}')
        set_seed(seed)
        torch.set_grad_enabled(False)
        set_reproducibility(args.reproduce, seed)
        return dp_mesh.size(), dp_mesh.get_local_rank(), dist.get_node_local_rank()
    elif mode == 'ddp':
        # Initialize distributed environment. We set a long timeout for unbalanced generation.
        dist.init_process_group("nccl", timeout=timeout)
        # We use DDP separately in each node to avoid between-node communication. So we need calculate the
        # world size and rank manually. We assume that each node has the same number of GPUs.
        world_size = args.num_nodes * torch.cuda.device_count()
        rank = dist.get_rank() + args.node_index * torch.cuda.device_count()
        device = rank % torch.cuda.device_count()

    elif mode == 'deepspeed':
        import deepspeed
        if args.use_hf:
            # 当使用 huggingface 推理时启用 deepspeed, 那么要求 pp_size > 1 且可以整除 8. 并且只让 pp0 运行主函数,
            # 其他进程 hold.
            deepspeed.init_distributed(timeout=get_nccl_timeout())

            pp_size = args.pp_size
            assert 8 % pp_size == 0, f"pp_size must be a divisor of 8, but got {pp_size}."
            assert pp_size > 1, f"pp_size must be greater than 1 when using hf deepspeed inference, but got {pp_size}."

            world_size = dist.get_world_size() // pp_size
            rank = dist.get_rank() // pp_size
            device = dist.get_rank() % torch.cuda.device_count()

        else:
            if timeout is not None:
                deepspeed.init_distributed(timeout=timeout)
            else:
                deepspeed.init_distributed()

            world_size = dist.get_world_size()
            rank = dist.get_rank()  # Rank of the current process in the cluster.
            device = rank % torch.cuda.device_count()  # Device of the current process in current node.

    else:
        world_size = 1
        rank = 0
        device = 0

    # Set current device for the current process.
    torch.cuda.set_device(device)
    # Disable gradients
    torch.set_grad_enabled(False)
    # Set reproducibility
    set_reproducibility(args.reproduce, args.global_seed)

    return world_size, rank, device


def setup_ptm_initialize():
    from megatron.initialize import initialize_megatron
    from megatron import get_args
    initialize_megatron(extra_args_provider=add_ptm_args)
    args = get_args()

    args.is_zero_stage_3 = True # only support stage3
    args.iteration = None # adapting load_checkpoint
    args = sanity_check_args(args) # set default values

    # Convert Namespace to EasyDict for uniform access
    args = EasyDict(vars(args))

    return args


def ceil_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


def to_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        else:
            return value.reshape(-1).tolist()
    else:
        return value


class BaseSampler(object):
    def __init__(self, args, model_dict, ckpt_path=None, rank=0, world_size=1, device=0, logger=None):
        if logger is None:
            from loguru import logger
        self.logger = logger
        self.args = args
        self.model_dict = model_dict
        self.ckpt_path = ckpt_path
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # For evaluation
        self._metric_dict = {}
        self.valid_metrics = set()
        self.score_models_loaded = False

        self.autocast_dtype = PRECISION_TO_TYPE[args.autocast_dtype]
        self.autocast_enabled = self.autocast_dtype != torch.float32

        # when use ptm inference, only tp_rank == 0 and pp_rank == last can save image
        self.save_image_flag = True
        if self.args.use_ptm:
            from megatron import mpu
            self.save_image_flag = mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage()
            print(f'global_rank:{torch.distributed.get_rank()}, self.save_image_flag:{self.save_image_flag}')

    @classmethod
    def from_pretrained(cls, ckpt_path, rank=0, world_size=1, device=0, logger=None, extra_args=None):
        """
        Initialize the sampling pipeline.

        Args:
            ckpt_path (str or pathlib.Path): The checkpoint path.
            rank (int): The rank for distributed inference. Default is 0.
            world_size (int): The world size for distributed inference. Default is 1.
            device (int): The device for inference. Default is 0.
            logger (logging.Logger): The logger for the inference pipeline. Default is None.
            extra_args (list): Extra arguments append to argv before parse args. Default is None.
        """
        if logger is None:
            from loguru import logger

        # ======================== Get the checkpoint path =======================
        if ckpt_path:
            ckpt_path = Path(ckpt_path)
            shard_ckpt_path = None
            if ckpt_path.is_dir():
                if (ckpt_path / "checkpoints/latest").exists():
                    latest_name = (ckpt_path / "checkpoints/latest").read_text().strip()
                    ckpt_path = ckpt_path / "checkpoints" / latest_name
                elif (ckpt_path / "latest").exists():
                    latest_name = (ckpt_path / "latest").read_text().strip()
                    ckpt_path = ckpt_path / latest_name
                try:
                    ckpt_path = next(ckpt_path.glob("*_model_states.pt"))
                except StopIteration:
                    # Check if is HF format
                    try:
                        shard_ckpt_path = next(ckpt_path.glob("pytorch_model-*.bin"))
                    except StopIteration:
                        raise FileNotFoundError(f"No model state found in the directory: {ckpt_path}")
            if shard_ckpt_path is not None:
                pass
            elif ckpt_path.is_file():
                assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
            else:
                raise FileNotFoundError(f"Invalid checkpoint path: {ckpt_path}")

        # ============================= Load the config ===========================
        args = cls.load_config(ckpt_path, logger, extra_args)

        # =========================== Build main model ===========================
        if ckpt_path is None and not args.no_load_model:
            raise ValueError("Missing --ckpt for evaluation or sampling.")

        logger.info("Building model...")
        factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
        if args.get("use_hf"):
            kwargs = dict(ckpt_path=ckpt_path)
        else:
            kwargs = {}
        model, model_settings = build_model(args, logger=logger, **factor_kwargs, **kwargs)

        logger.info(f"Loading model from: {ckpt_path} ({args.load_key})")
        if args.no_load_model:
            logger.info("Skip loading model.")
        elif args.get("use_hf"):
            # HF weights are already loaded when building model.
            pass
        else:
            cls.load_state_dict(model, ckpt_path, args.load_key, weights_only=args.weights_only)
        logger.info(f"Loaded.")
        model.requires_grad_(False)
        model.eval()
        model_dict = dict(model=model, model_settings=model_settings)

        if args.reproduce:
            model.enable_deterministic()

        # =========================== Build extra model ===========================
        model_dict = cls.build_extra_model(args, model_dict, factor_kwargs, logger)

        return cls(
            args=args,
            model_dict=model_dict,
            ckpt_path=ckpt_path,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )

    @classmethod
    def ptm_from_pretrained(cls, args, logger = None):
        import deepspeed
        from megatron import print_rank_0, mpu
        from megatron.initialize import initialize_megatron
        from megatron.checkpointing import load_checkpoint
        from deepspeed.runtime.utils import see_memory_usage
        from megatron.utils import unwrap_model
        from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
        from megatron.model import DistributedDataParallel as LocalDDP
        from megatron.model import Float16Module
        from megatron.optimizer import get_megatron_optimizer
        from megatron.checkpointing import save_checkpoint
        from hymm.ptm.training import print_datetime

        # =========================== Build main model ===========================
        see_memory_usage(f"Before Building Model", force=True)
        with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                                    remote_device=None if args.remote_device == 'none' else args.remote_device,
                                    config_dict_or_path=args.deepspeed_config,
                                    enabled=args.zero_stage == 0,
                                    mpu=mpu):
            factor_kwargs = {'device': torch.device("cuda", args.local_rank), 'dtype': PRECISION_TO_TYPE[args.precision]}
            # print_rank_0(f"before build_model, args:{args}, dict_args:{dict_args}")
            model, model_settings = build_model(args, **factor_kwargs)
        see_memory_usage(f"After Building Model", force=True)

        # =========================== Load ckpt, don't init optim ===========================
        if not args.no_pipeline_parallel and args.mt_pipeline_parallel:
            mt_pipeline_parallel = True
        elif args.mmp_encoder_parallel:
            mt_pipeline_parallel = True
        else:
            mt_pipeline_parallel = False
        unwrapped_model = unwrap_model([model],
                (torchDDP, LocalDDP, Float16Module))
        params = get_megatron_optimizer(unwrapped_model)
        # inference mode, optimizer is DummyOptim, don't occupy device mem
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            infer_model_parameters=params if args.no_init_optim else None,
            model_parameters=params if not args.no_init_optim else None,
            args=args,
            lr_scheduler=None,
            mpu=mpu if args.no_pipeline_parallel or mt_pipeline_parallel else None
        )
        see_memory_usage(f"After deepspeed.initialize", force=True)

        if args.no_load_model:
            iteration = 0
            logger.info("Skip loading model.")
        else:
            iteration = load_checkpoint([model], optimizer, lr_scheduler)
            see_memory_usage(f"After load_checkpoint", force=True)

        if args.save_after_load:
            print_datetime('Saving the model after loads it!')
            save_checkpoint(args.iteration+1, model, optimizer, lr_scheduler, save_tensors_mode=args.save_tensors_mode)
            print_datetime('models! saved!')
        
        # set eval mode
        module = model.module
        module.requires_grad_(False)
        module.eval()

        # =========================== Build extra model ===========================
        model_dict = dict(model=module, model_settings=model_settings)
        model_dict = cls.build_extra_model(args, model_dict, factor_kwargs, logger)
        see_memory_usage(f"After build_extra_model", force=True)

        dp_rank = mpu.get_data_parallel_rank()
        dp_world_size =  mpu.get_data_parallel_world_size()
        return cls(
            args=args,
            model_dict=model_dict,
            ckpt_path=args.load,
            rank=dp_rank,
            world_size=dp_world_size,
            device=args.local_rank,
            logger=logger,
        )


    @staticmethod
    def load_state_dict(model, ckpt_path, load_key=None, weights_only=False):
        if isinstance(load_key, str):
            state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage,
                                    weights_only=weights_only)
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            elif load_key is None:
                pass
            else:
                raise KeyError(f"Key '{load_key}' not found in the checkpoint. Existed keys: {state_dict.keys()}")
            model.load_state_dict(state_dict)

        elif isinstance(load_key, dict):
            m, u = load_state_dict(
                model, str(ckpt_path),
                load_llm_to_mllm=load_key.get('load_key_load_llm_to_mllm', False),
                expand_keys=load_key.get('load_key_expand_keys'),
                load_prepend_key=load_key.get('load_key_load_prepend_key'),
                load_prepend_key_dict=load_key.get('load_key_load_prepend_key_dict'),
            )
            print(f"loaded pretrained checkpoint,\n missing keys: {m}\nunexpected keys: {u}")

    @staticmethod
    def load_config(ckpt_path, logger, extra_args=None):
        def get_args_from_env():
            import shlex
            env_params = os.getenv('MULTI_MODA_PARAMS', '')
            if not env_params:
                return []
            # 使用shlex.split来正确解析参数（处理引号、空格等）
            args = shlex.split(env_params)
            return args

        use_ext_model_ = should_use_external_model()
        if use_ext_model_:
            argv = get_args_from_env()
            print(f"get_args_from_env:{argv}")
        else:
            argv = sys.argv[1:]
        if "--config-path" not in argv:
            if ckpt_path is None:
                raise ValueError("Missing config specification. Please provide --ckpt or --config-path.")
            ckpt_config_path = ckpt_path.parents[2] / "config.yaml"
            if not ckpt_config_path.exists():
                raise ValueError(f"Missing config.yaml in ckpt directory: {ckpt_path.parents[2]}.\n"
                                 f"Please explicitly specify the config path by --config-path.")
            argv.extend(["--config-path", str(ckpt_config_path)])
            argv.extend(default(extra_args, []))
        args = parse_args(argv)
        # Format args for better readability
        args_str = "\nConfiguration:"
        for key, value in vars(args).items():
            args_str += f"\n  {key}: {value}"
        logger.info(args_str)
        return args

    def get_exp_dir_and_ckpt_id(self):
        if self.ckpt_path is None:
            raise ValueError("The checkpoint path is not provided.")

        ckpt_path = Path(self.ckpt_path)
        if ckpt_path.parents[1].name == "checkpoints":
            # It should be a standard checkpoint path. We use the parent directory as the default save directory.
            exp_dir = ckpt_path.parents[2]
        else:
            raise ValueError(
                f"We cannot infer the experiment directory from the checkpoint path: {ckpt_path}. "
                f"It seems that the checkpoint path is not standard. Please explicitly provide the "
                f"save path by --save-dir (when using --csv) or --sample-save-base (when using --testsets)."
            )
        return exp_dir, ckpt_path.parent.name

    @staticmethod
    def parse_image_size(image_size, align=1, base_size=1024):
        if isinstance(image_size, str):
            if image_size == "auto":
                image_size = [base_size, base_size]
            else:
                image_size = list(map(int, image_size.split("x")))
        if isinstance(image_size, int):
            image_size = [image_size]
        if not isinstance(image_size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {image_size}.")
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        if len(image_size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {image_size}.")
        # Align the size to the given value
        if isinstance(align, int):
            align = [align, align]
        if not isinstance(align, (list, tuple)):
            raise ValueError(f"Align must be an integer or (height, width), got {align}.")
        if len(align) == 1:
            align = [align[0], align[0]]
        if len(align) != 2:
            raise ValueError(f"Align must be an integer or (height, width), got {align}.")
        image_size = [ceil_to(s, a) for s, a in zip(image_size, align)]
        return image_size

    def get_metric_save_path(self, metric_type, image_size=None):
        if self.args.metric_save_path is None:
            exp_dir, ckpt_id = self.get_exp_dir_and_ckpt_id()
            save_dir = exp_dir / "evaluation"
            if image_size is not None:
                image_size = to_2tuple(image_size)
                image_size_str = f"{image_size[0]}x{image_size[1]}"
                save_file = f"{ckpt_id}_{metric_type}_{image_size_str}{self.args.metric_save_file_suffix}.json"
            else:
                save_file = f"{ckpt_id}_{metric_type}{self.args.metric_save_file_suffix}.json"
            save_path = str(save_dir / save_file)
            self.logger.info(f"Set metric_save_file to {save_path} based on the checkpoint path.")
        else:
            save_path = self.args.metric_save_path
        if not save_path.endswith(".json"):
            raise ValueError(f"args.metric_save_path should be a json file, but got {save_path}")

        return save_path

    def get_sample_dir_suffix(self, task=None, testset=None, image_size=None, load_key=None, segments=None, rerank=None):
        final_segments = []
        final_segments += [task] if task is not None else []

        if image_size is not None and all(item>0 for item in ensure_list(image_size)):
            h, w = self.parse_image_size(image_size)
            final_segments += [f"{h}x{w}"]
        elif ensure_list(image_size)[0] < 0:
            final_segments += ["dynamic"]
        final_segments += [default(load_key, self.args.load_key if isinstance(self.args.load_key, str) else "module")]
        # Put testset after load_key, so the order of the segments is aligned with the image_save_base in evaluator
        final_segments += [testset] if testset is not None else []
        final_segments += default(segments, [])
        final_segments += [f"rerank{rerank}"] if rerank is not None else []

        fixed_suffix = "_".join(final_segments)
        return fixed_suffix

    def get_sample_save_dir(self, task=None, testset=None, image_size=None, segments=None, rerank=None, subdir="samples"):
        if self.args.sample_save_path is None:
            if self.args.sample_save_base is not None:
                save_base = Path(self.args.sample_save_base)
                if testset is not None:
                    save_base = save_base / testset.replace("@@", "_").replace("=", "_")
                    self.logger.info(f"Set save_base to {save_base}.")
            else:
                exp_dir, ckpt_id = self.get_exp_dir_and_ckpt_id()
                fixed_suffix = self.get_sample_dir_suffix(task=task, testset=testset, image_size=image_size, segments=segments, rerank=rerank)
                save_base = exp_dir / subdir / f"{ckpt_id}_{fixed_suffix}{self.args.sample_save_path_suffix}"
                self.logger.info(f"Set save_base to {save_base} based on the checkpoint path.")
        else:
            save_base = Path(self.args.sample_save_path)
            save_base = save_base.parent / f"{save_base.name}{self.args.sample_save_path_suffix}"

        return save_base

    @staticmethod
    def get_default_sample_save_paths(num_images, prompt, save_dir=None):
        today = datetime.datetime.now().strftime("%Y%m%d")
        save_path = Path(default(save_dir, f"vis/images/output/{today}"))
        if save_path.suffix == "":
            # We treat it as a directory
            start_id = get_next_available_save_id(save_path)
            save_paths = []
            for i in range(num_images):
                save_stem = f"{i + start_id}_{prompt}"
                save_paths.append(save_path / (save_stem[:240] + ".png"))
        else:
            if num_images == 1:
                save_paths = [save_path]
            else:
                save_paths = [
                    save_path.parent / f"{save_path.stem}({i + 1}){save_path.suffix}" for i in range(num_images)
                ]

        return save_paths

    @staticmethod
    def save_batch_data(results: Union[pd.DataFrame, List[Dict[str, Any]]], save_path):
        save_path = safe_file(save_path)
        if not isinstance(results, pd.DataFrame):
            results = pd.DataFrame(results)
        results.to_csv(save_path, index=False, mode='a', header=not save_path.exists())
        return save_path

    @staticmethod
    def save_batch_image(images, save_names, prompts=None, prompts_save_path=None):
        """
        images can be:
        * torch.Tensor, shape (B, C, H, W)
        * numpy.ndarray, values in interval [0, 1], shape (H, W, C) or (B, H, W, C)
        * a list of PIL.Image.
        """
        if len(images) != len(save_names):
            raise ValueError(
                f"Length of images ({len(images)}) should be equal to length of save_names ({len(save_names)})."
            )

        if isinstance(images, torch.Tensor):
            # Tensor -> numpy.ndarray
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        if isinstance(images, np.ndarray):
            # numpy.ndarray -> PIL.Image
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            if images.shape[-1] == 1:
                # special case for grayscale (single channel) images
                images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
            else:
                images = [Image.fromarray(image) for image in images]

        for i, (image, save_name) in enumerate(zip(images, save_names)):
            image.save(safe_file(save_name))

        # Save prompts
        if prompts is not None:
            if not isinstance(prompts, pd.DataFrame):
                prompts = pd.DataFrame({"path": [Path(save_name).name for save_name in save_names], "prompt": prompts})
            BaseSampler.save_batch_data(prompts, prompts_save_path)

    @staticmethod
    def prepare_prompts(prompt, negative_prompt=None, **kwargs):
        prompt_embeds = kwargs.get("prompt_embeds", None)
        attention_mask = kwargs.get("attention_mask", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
        negative_attention_mask = kwargs.get("negative_attention_mask", None)
        if prompt is None:
            # prompt_embeds, attention_mask, negative_prompt_embeds and negative_attention_mask should not be None
            # pipeline will help to check this
            prompt = None
            negative_prompt = None
            batch_size = prompt_embeds.shape[0]
        else:
            # prompt_embeds, attention_mask, negative_prompt_embeds and negative_attention_mask should be None
            # pipeline will help to check this
            if isinstance(prompt, str):
                batch_size = 1
                prompt = [prompt]
            elif isinstance(prompt, (list, tuple)):
                batch_size = len(prompt)
            else:
                raise ValueError(f"Prompt must be a string or a list of strings, got {prompt}.")

            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
        return dict(prompt=prompt,
                    negative_prompt=negative_prompt,
                    prompt_embeds=prompt_embeds,
                    attention_mask=attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_attention_mask=negative_attention_mask), batch_size

    @staticmethod
    def prepare_seed(seed, batch_size, num_sample_per_prompt):
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 1_000_000) for _ in range(batch_size * num_sample_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for _ in range(batch_size) for i in range(num_sample_per_prompt)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) + j for i in range(batch_size) for j in range(num_sample_per_prompt)]
            elif len(seed) == batch_size * num_sample_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt({batch_size}) or "
                    f"batch_size * num_sample_per_prompt ({batch_size} * {num_sample_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        return seeds

    def get_infer_kwargs(self, safe=False):
        return {}

    def build_sample_dataloader(
        self,
        data_source,
        save_template=None,
        batch_size=1,
        seed_type="auto",
        seed=123456,
        skip_exist=False,
        dataset_kwargs=None,
        collate_fn=None,
    ):
        if isinstance(data_source, DataLoader):
            dataloader = data_source
        else:
            if isinstance(batch_size, (list, tuple)):
                assert len(batch_size) == 1, f"Expected a single batch size, got {batch_size}"
                batch_size = batch_size[0]
            if isinstance(data_source, str):
                dataset = CSVDataset(data_source,
                                     save_template=save_template,
                                     seed_type=seed_type,
                                     seed=seed,
                                     skip_exist=skip_exist,
                                     logger=self.logger,
                                     **default(dataset_kwargs, {}),
                                     )
            elif isinstance(data_source, Dataset):
                dataset = data_source
            elif callable(data_source):
                dataset = data_source(save_template=save_template)
            else:
                raise TypeError(f"Invalid data_source type: {type(data_source)}")

            # ptm inference need to avoid different sample numbers on each dp rank
            sampler = DistributedSamplerFix(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
                add_extra_samples=True if (self.args.use_ptm or self.args.launcher == 'pure_torch') and self.args.eval_data_repeat_times == 0 else False,
                repeat_times=self.args.eval_data_repeat_times,
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=False, collate_fn=collate_fn)
        return dataloader

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def rerank_wrapper(self, fn, rerank, input_name, inputs, seeds, clip_prompt=None, **kwargs):
        if isinstance(inputs, list):
            if isinstance(seeds, int):
                seeds = [seeds]
            assert len(inputs) == len(seeds), (
                f"Length of prompts ({len(inputs)}) should be equal to length of seeds ({len(seeds)})."
            )
        clip_prompt = default(clip_prompt, inputs)

        rerank_selected_samples = None
        prompts = None
        out_seeds = None
        best_clip_score = None
        for i in range(rerank):
            self.logger.info(f"Rerank {i + 1}/{rerank}")
            cur_seeds = [seed + i for seed in seeds]    # make seed different
            outputs = fn(**{input_name: inputs}, seed=cur_seeds, **kwargs)
            samples = outputs["samples"].float()
            prompts = outputs.get("prompts")
            out_seeds = outputs.get("seeds")
            if i == 0:
                rerank_selected_samples = samples
                best_clip_score = self.rerank_clip_score_metric.calculate_clip_score(prompts=clip_prompt, images=samples)
                self.logger.debug(f"Best clip_score: {best_clip_score.tolist()}")
            else:
                cur_clip_score = self.rerank_clip_score_metric.calculate_clip_score(prompts=clip_prompt, images=samples)
                better_idx = cur_clip_score > best_clip_score
                rerank_selected_samples = torch.where(better_idx.unsqueeze(1).unsqueeze(1).unsqueeze(1), samples,
                                                      rerank_selected_samples)
                best_clip_score[better_idx] = cur_clip_score[better_idx]
                self.logger.debug(f"Best clip score: {best_clip_score.tolist()}")
        final_samples = self.numpy_to_pil(self.pt_to_numpy(rerank_selected_samples))
        return {
            "samples": final_samples,
            "prompts": prompts,
            "seeds": out_seeds,
        }

    def load_clip_score_model(self):
        """ For rerank """
        from hymm.metrics.clip_score.metric import CLIPScoreMetric
        from hymm.constants import CLIP_MODEL_PATH

        self.rerank_clip_score_metric = CLIPScoreMetric(clip_model_path=CLIP_MODEL_PATH)
        self.rerank_clip_score_metric.load_model(self.logger)

    # input_batch_dict, key: argument name of self.predict, value: the argument's corresponding key in the batch fetched from dataloader
    def batch_sample(self,
                     dataloader,
                     input_batch_dict=None,
                     rerank=1,
                     **kwargs):
        infer_kwargs = self.get_infer_kwargs()
        for key, value in infer_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        # Rerank Prepare
        rerank_enabled = False
        if rerank > 1:
            rerank_enabled = True
            self.load_clip_score_model()
        else:
            self.rerank_clip_score_metric = None

        # Start sampling
        total_batches = len(dataloader)
        for batch_index, batch in enumerate(dataloader, start=1):
            batch: Dict[str, Any]
            self.logger.info(f"Batch {batch_index}/{total_batches}")
            # Adjust max_width according to your terminal width
            self.logger.info(f"\n{batch_data_repr(batch, max_width=150)}")

            inputs = {}
            if "type" in batch:
                inputs[batch["type"][0]] = batch["input"]
            if "seed" in batch:
                inputs["seed"] = batch["seed"]
            for k, v in default(input_batch_dict, {}).items():
                if v in batch:
                    inputs[k] = batch[v]

            kwargs = default(kwargs, {})
            # override kwargs with inputs
            for k in inputs.keys():
                if k in kwargs:
                    del kwargs[k]
            
            if rerank_enabled:
                outputs = self.rerank_wrapper(
                    fn=self.predict,
                    rerank=rerank,
                    input_name="prompt",
                    inputs=inputs['prompt'],
                    seeds=inputs['seed'],
                    output_type="pt",
                    **kwargs,
                )
            else:
                outputs = self.predict(
                    **inputs,
                    verbose=1,
                    **kwargs,
                )
            final_samples = outputs["samples"]
            prompts = outputs.get("prompts") if kwargs.get('save_prompts', False) else None

            if self.save_image_flag:
                save_names = [
                    batch["save_path"][i // self.args.num_sample_per_prompt].format(
                        i % self.args.num_sample_per_prompt
                    )
                    for i, sample in enumerate(final_samples)
                ]

                # Save prompts when prompts is not None
                prompts_save_path = Path(save_names[0]).parent / f"prompts_{self.rank}.csv"
                self.save_batch_image(final_samples, save_names, prompts, prompts_save_path)

                # save other images during sampling, e.g, cropped face, ground truth face, etc.
                for k, v in outputs.items():
                    if k.startswith("save_") and isinstance(v, torch.Tensor):
                        self.logger.info(f"Save {k}: {v.shape}")
                        k_wo_save = k.replace("save_", "")
                        k_save_names= []
                        for name in save_names:
                            k_save_names.append(name.replace('.png', f"_{k_wo_save}.png"))
                        self.save_batch_image(v, k_save_names)
        # wait for all worker
        if dist.is_initialized():
            dist.barrier()

    # input_batch_dict, key: argument name of self.predict, value: the argument's corresponding key in the batch fetched from dataloader
    def batch_sample_siglip(self,
                     dataloader,
                     input_batch_dict=None,
                     **kwargs):
        infer_kwargs = self.get_infer_kwargs()
        for key, value in infer_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        # Start sampling
        total_batches = len(dataloader)
        for batch_index, batch in enumerate(dataloader, start=1):
            batch: Dict[str, Any]
            self.logger.info(f"Batch {batch_index}/{total_batches}")
            # Adjust max_width according to your terminal width
            self.logger.info(f"\n{batch_data_repr(batch, max_width=150)}")

            inputs = {}
            if "type" in batch:
                inputs[batch["type"][0]] = batch["input"]
            if "seed" in batch:
                inputs["seed"] = batch["seed"]
            for k, v in default(input_batch_dict, {}).items():
                inputs[k] = batch[v]

            outputs = self.predict(
                **inputs,
                verbose=1,
                **kwargs,
            )
            final_latents = outputs["latents"]

            prompts = outputs.get("prompts")

            pt_save_names = [
                batch["save_path"][i // self.args.num_sample_per_prompt].format(
                    i % self.args.num_sample_per_prompt,
                    "pt"
                )
                for i, _ in enumerate(final_latents)
            ]

            csv_save_names = [
                batch["save_path"][i // self.args.num_sample_per_prompt].format(
                    i % self.args.num_sample_per_prompt,
                    "csv"
                )
                for i, _ in enumerate(final_latents)
            ]

            for id, seed, latent, prompt, pt_save_name, csv_save_name in zip(batch["id"], batch["seed"], final_latents, prompts, pt_save_names, csv_save_names):
                torch.save(latent, safe_file(pt_save_name))
                data = [{
                    "index": id.item(),
                    "seed": seed.item(),
                    "prompt": prompt,
                    "height":16,
                    "width": 16,
                    "feature_path": pt_save_name
                }]
                with open(safe_file(csv_save_name), 'w', newline='\n', encoding='utf-8-sig') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)

        # wait for all worker
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def build_extra_model(args, model_dict, factor_kwargs, logger=None):
        return model_dict

    @torch.no_grad()
    def predict(self, **kwargs):
        """
        returns: {"samples": samples}, where samples can be a torch.Tensor, numpy.ndarray, or a list of PIL.Image
        """
        raise NotImplementedError()

    # =====================================================================================================
    # For evaluation
    # =====================================================================================================
    def initialize_scores(self, scores):
        from hymm.metrics import load_metric

        metric_dict = {}
        assert len(scores) > 0, "Must specify at least one metric to evaluate."
        for score in scores:
            kwargs = {}
            if score == "mmlu_bench":
                kwargs = {"tokenizer": self.model_dict["tokenizer"].tokenizer}
            metric_dict[score] = load_metric(score, **kwargs)
        self._metric_dict = metric_dict

    @property
    def metric_dict(self):
        if not self._metric_dict:
            raise ValueError("No metric is initialized. Call `evaluator.initialize_scores(scores)` first.")
        return self._metric_dict

    def load_score_models(self, size=256):
        """Load score models to evaluate."""
        for metric_name, metric in self.metric_dict.items():
            if not hasattr(metric, 'max_size') or size <= metric.max_size:
                metric.load_model(self.logger)
        self.score_models_loaded = True

    def release_score_models(self, size=256):
        """Release score models to save memory."""
        for metric_name, metric in self.metric_dict.items():
            if not hasattr(metric, 'max_size') or size <= metric.max_size:
                metric.release_model()
        self.score_models_loaded = False
        # Clear cuda cache
        torch.cuda.empty_cache()

    def get_dataloader(self, dataset, batch_size):
        # ptm inference need to avoid different sample numbers on each dp rank
        dataset_sampler = DistributedSamplerFix(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
            add_extra_samples=True if (self.args.use_ptm or self.args.launcher == 'pure_torch') and self.args.eval_data_repeat_times == 0 else False,
            repeat_times=self.args.eval_data_repeat_times,
        )
        if self.args.eval_data_repeat_times > 0:
            self.logger.info(f"Repeat dataset {self.args.eval_data_repeat_times} times for evaluation.")
        dataloader = DataLoader(
            dataset=dataset,
            sampler=dataset_sampler,
            pin_memory=True,
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
            batch_size=batch_size,
        )
        return dataloader

    def get_eval_datasets(self, size):
        """ Get evaluation datasets according to the size. """
        valid_datasets = []
        for metric_name, metric in self.metric_dict.items():
            if not hasattr(metric, 'max_size') or size <= metric.max_size:
                valid_datasets.append(metric.dataset_name)
        return valid_datasets

    @torch.no_grad()
    def eval(
            self,
            image_size,
            batch_size,
            save_path,
            rerank=1,
            image_save_base=None,
            extra_save_info=None,
            release_models=False,
            input_processor=None,
            dataset_kwargs=None,
            run_fn=None,
            run_fn_kwargs=None,
            data_save_base=None,
            **kwargs,
    ):
        """
        Evaluate the model on the specified metrics.

        Parameters
        ----------
        image_size : int or tuple
            The size of the images to generate.
        batch_size : int
            The batch size for evaluation.
        save_path : str or pathlib.Path
            The path to save the evaluation results.
        rerank : int, optional
            The number of rerank times. Default is 1.
        image_save_base : str or pathlib.Path, optional
            The base path to save the images. If None, images will not be saved. Default is None.
        extra_save_info : dict, optional
            Extra information to save with the evaluation results. Default is None.
        kwargs : dict
            Other keyword arguments passed to self.predict() for evaluation.
        """
        self.logger.info(f"Start evaluation and save to {save_path}")
        run_fn = default(run_fn, self.predict)
        run_fn_kwargs = default(run_fn_kwargs, {})

        # Sanity check for mandatory arguments
        require_save_dir = any(
            'save_file' in metric.compute_metrics_required_args
            for metric in self._metric_dict.values()
        )
        save_dir = data_save_base if data_save_base is not None else image_save_base
        if require_save_dir:
            assert save_dir is not None, \
                f"data_save_base is required by metrics: {list(self._metric_dict.keys())}"
        if save_dir is not None:
            # Save intermediate results (images, answers) to save_dir if save_dir is provided
            self.logger.info(f"Evaluation intermediate results will be saved to {save_dir}")

        # Load score models on demand
        hw = to_2tuple(image_size)
        if not self.score_models_loaded:
            self.load_score_models(min(hw))

        # Check batch_size, and convert it to a single integer
        if isinstance(batch_size, (list, tuple)):
            assert len(batch_size) == 1, f"Expected a single batch size, got {batch_size}"
            batch_size = batch_size[0]

        # Check extra_save_info to avoid saving error and convert it to a dictionary
        extra_save_info = default_dtype(extra_save_info, {})

        # Load datasets. A dataset is valid if its max_size is greater than or equal to the current target size.
        # Remove duplicates for saving computation.
        valid_dataset_names = sorted(list(set([
            metric.dataset_name
            for metric_name, metric in self.metric_dict.items()
            if not hasattr(metric, 'max_size') or min(hw) <= metric.max_size
        ])))
        dataloaders = []
        for dataset_name in valid_dataset_names:
            dataset = load_dataset(dataset_name, **default(dataset_kwargs, {}))
            if (collate_fn := kwargs.get('collate_fn')) is not None:
                dataset.collate_fn = collate_fn
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"{dataset_name} dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, dataset_name))

        # Rerank Prepare
        rerank_enabled = False
        if rerank > 1:
            rerank_enabled = True
            from hymm.metrics.clip_score.metric import CLIPScoreMetric
            from hymm.constants import CLIP_MODEL_PATH
            rerank_clip_score_metric = CLIPScoreMetric(clip_model_path=CLIP_MODEL_PATH)
            rerank_clip_score_metric.load_model(self.logger)

        # ----------------------------------------------------------------------
        # Prepare valid_metrics first.
        for dataloader, dataset, dataset_name in dataloaders:
            for metric_name, metric in self.metric_dict.items():
                if metric.dataset_name == dataset_name:
                    self.valid_metrics.add(metric_name)
        predictions = {}
        # Do prediction
        for dataloader, dataset, dataset_name in dataloaders:
            total_batches = len(dataloader)
            self.logger.info(f"*************************************")
            self.logger.info(f"    Evaluation on {dataset_name}.    ")
            self.logger.info(f"*************************************")
            print(f"[rank{self.rank}] Total batches: {total_batches}, dp samples: {len(dataloader.sampler)}")

            metric_input_key = getattr(dataset, "metric_input_key", "images")
            run_fn_kwargs_2 = getattr(dataset, "run_fn_kwargs", {})
            predictions[dataset_name] = {
                "save_dir": save_dir,
                "data": [],
                "metric_input_key": metric_input_key,
            }

            for batch_index, batch in enumerate(dataloader, start=1):
                batch: Dict[str, Any]
                self.logger.info(f"Batch {batch_index}/{total_batches}")

                inputs_dict = {}

                # for single input; key is data_type, value is input
                is_single_input = "type" in batch
                if is_single_input:
                    data_type = batch.pop('type')[0]        # data_type = prompt/label
                    inputs = batch.pop('input')             # prompt list or label tensor
                    inputs_dict[data_type] = inputs

                # for multiple inputs; key is mapped by input_batch_dict, value is in batch
                if "input_batch_dict" in kwargs:
                    for k, v in default(kwargs["input_batch_dict"], {}).items():
                        if v in batch:
                            inputs_dict[k] = batch[v]
                        elif v in inputs_dict:
                            inputs_dict[k] = inputs_dict[v]
                        else:
                            raise ValueError(f"Key {v} not found in batch or inputs_dict.")

                if "seeds" in batch:
                    inputs_dict["seed"] = batch["seeds"]

                # The input_processor is applied to allow one to change the input data before passing it to the model.
                # For example, one can recaption the prompt in the input_processor.
                if input_processor is not None:
                    inputs_dict = input_processor(self, inputs_dict)

                if rerank_enabled and not (dataset_name in ["mmlu_bench", "mmlu_pro_bench"]):
                    assert "type" in batch, "rerank only support single input; has not been tested for multiple inputs"
                    rerank_selected_samples = None
                    best_clip_score = None
                    for i in range(rerank):
                        self.logger.info(f"Rerank {i+1}/{rerank}")
                        outputs = run_fn(
                            **{data_type: inputs_dict[data_type]},
                            size=image_size,
                            seed=[seed+i for seed in batch["seeds"]], # make seed different
                            output_type="pt",
                            verbose=1,
                            **kwargs,
                            **run_fn_kwargs,
                        )
                        samples = outputs["samples"].float()
                        if i == 0:
                            rerank_selected_samples = samples
                            best_clip_score = rerank_clip_score_metric.calculate_clip_score(prompts=inputs, images=samples)
                        else:
                            cur_clip_score = rerank_clip_score_metric.calculate_clip_score(prompts=inputs, images=samples)
                            better_idx = cur_clip_score > best_clip_score
                            rerank_selected_samples = torch.where(better_idx.unsqueeze(1).unsqueeze(1).unsqueeze(1), samples, rerank_selected_samples)
                            best_clip_score[better_idx] = cur_clip_score[better_idx]
                    final_samples = rerank_selected_samples
                else:
                    outputs = run_fn(
                        **inputs_dict,
                        size=image_size,
                        output_type="pt",
                        **kwargs,
                        **run_fn_kwargs,    # additional kwargs for all datasets
                        **run_fn_kwargs_2,  # additional kwargs for current dataset
                    )
                    final_samples = outputs["samples"]
                    if metric_input_key == "images":
                        final_samples = final_samples.float()
                    elif metric_input_key == "pred_logits":  # for mmlu_bench metric
                        # print(f"outputs: {outputs}")
                        assert "first_pred_token_logits" in outputs, f"Output of 'first_pred_token_logits' is required for mmlu_bench metric computing."
                        final_samples = outputs["first_pred_token_logits"]

                    # Finish status determined by flag
                    if "finished_by_dummy" in outputs and outputs["finished_by_dummy"]:
                        break

                proc_inputs = batch
                if is_single_input:
                    proc_inputs[data_type] = inputs     # We always use the original inputs for metrics
                proc_inputs[metric_input_key] = final_samples

                # Skip dummy batch processing
                if "is_dummy" in batch and batch["is_dummy"].all():
                    assert len(batch["is_dummy"]) == 1, "Only support batch size 1 for dummy batch."
                    self.logger.info(f"All dummy batch (batch_id={batch_index}), skip processing.")
                    continue

                start_time = time.time()
                for metric_name, metric in self.metric_dict.items():
                    if metric.dataset_name == dataset_name:
                        metric.process(**proc_inputs)
                gen_time = time.time() - start_time
                self.logger.info(f"Metric process time: {gen_time}")

                if save_dir is not None:
                    if metric_input_key == "images":
                        save_names = [Path(save_dir) / dataset_name / f"{id_}.png" for id_ in batch["ids"]]
                        self.save_batch_image(final_samples, save_names)
                    elif metric_input_key == "answers":
                        for si, id_ in enumerate(batch["ids"]):
                            predictions[dataset_name]["data"].append({
                                "index": to_scalar(id_),
                                "seed": to_scalar(batch["seeds"][si]),
                                "answer": final_samples[si]
                            })
        # Don't do synchronization in dataloader loop, because distributed sampler
        # may not have the same number of batches in each process.
        # --------------------------------------------------------------------

        # Release score models to save memory
        if release_models:
            self.release_score_models(min(hw))
            torch.cuda.empty_cache()
        # Barrier to ensure all processes have finished the evaluation and have the same `self.valid_metrics`
        dist.barrier()
        self.logger.info(f"All processes have finished the evaluation; start gather results for self.valid_metrics: {self.valid_metrics}")

        # All gather answers
        for dataset_name, item in predictions.items():
            if item["save_dir"] is not None:
                if item["metric_input_key"] == "answers":
                    merged_data = [None for _ in range(self.world_size)]
                    torch.distributed.all_gather_object(merged_data, item["data"])
                    if self.rank == 0:
                        merged_data = sorted(list(chain(*merged_data)), key=lambda x: x["index"])
                        BaseSampler.save_batch_data(merged_data, item["save_dir"] / f"{dataset_name}_answer.csv")

        # Do synchronization here.
        metric_gather_results = {}
        for metric_name, metric in self.metric_dict.items():
            if metric_name in self.valid_metrics:
                self.logger.info(f"All gather results for {metric_name}")
                metric_gather_results[metric_name] = metric.all_gather_results()

        self.logger.info(f"All gather results for all metrics finished")
        results = []
        if self.rank == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            for metric_name, metric in self.metric_dict.items():
                if metric_name in self.valid_metrics:

                    metric_kwargs = {}
                    accept_save_file = 'save_file' in list(inspect.signature(metric.compute_metrics).parameters.keys())
                    if accept_save_file and save_dir is not None and hasattr(metric, "save_file_template"):
                        metric_kwargs['save_file'] = save_dir / metric.save_file_template.format(
                            timestamp.replace('-', '_'))
                        self.logger.info(f"[{metric_name}] Intermediate results will be saved to {metric_kwargs['save_file']}")

                    output = metric.compute_metrics(metric_gather_results[metric_name], **metric_kwargs)
                    if isinstance(output, tuple):
                        output = {"_": output}

                    for key, (value, count, *extra_outputs) in output.items():
                        suffix = "" if key == "_" else f"_{key}"
                        results.append({
                            "metric": f"{metric_name}{suffix}",
                            "value": value,
                            "count": count,
                            "rerank": rerank,
                            "timestamp": timestamp,
                            "extra": extra_save_info,
                        })
                        if len(extra_outputs) > 0:
                            # Metrics like VQAv2 return a dict to save some extra information
                            if 'extra_info_dict' in extra_outputs[0]:
                                results[-1]["extra_metric_stats"] = extra_outputs[0]['extra_info_dict']

            self.logger.info(results)

            accumulated_results = results[:]
            save_path = safe_file(save_path)
            if save_path.exists():
                with open(save_path, "r") as f:
                    ori_results = json.load(f)
                accumulated_results = ori_results + results

            save_to = safe_save_file(save_path, accumulated_results, save_fn=save_to_json)
            self.logger.info(f"Evaluation results saved to {save_to}")

        # gather finish, reset
        for metric_name, metric in self.metric_dict.items():
            metric.reset()
        self.valid_metrics.clear()

        # Only rank-0 returns the valid results
        return results


def x2image_interactive(input_fn, args, sampler, logger=None, **kwargs):
    """
    A wrapper for starting the interactive inference pipeline.
    """
    if logger is None:
        from loguru import logger
    while True:
        inputs = input_fn()
        if inputs is None:
            break

        # Determine the seed
        if args.seed_type in ["auto", "fixed"]:
            seed = args.seed
        elif args.seed_type == "random":
            seed = None
        else:
            raise ValueError(
                f"When evaluating `prompt`, `seed_type` must be one of ['auto', 'fixed', 'random'], "
                f"got {args.seed_type}."
            )
        # Start sampling
        outputs = sampler.predict(
            **inputs,
            size=args.sample_image_size,
            seed=seed,
            verbose=1,
            **kwargs,
        )
        samples = outputs["samples"]
        # Save the generated images
        if "prompt" in inputs:
            save_name = inputs["prompt"]
        elif "label" in inputs:
            save_name = str(inputs["label"])
        else:
            raise ValueError("Either `prompt` or `label` must be provided.")
        save_paths = sampler.get_default_sample_save_paths(len(samples), save_name, save_dir=args.sample_save_path)
        sampler.save_batch_image(samples, save_paths)
        logger.info(f"Save the generated image to: {save_paths}")


def text2image_interactive(args, sampler, logger=None, **kwargs):
    def input_fn():
        if args.prompt is None:
            # Ask for the next prompt
            inputs = input("Input prompt (`q` to quit): ")
            if inputs == "q":
                return None
            prompt = inputs
        else:
            prompt = args.prompt
            args.prompt = None
        return {'prompt': prompt}

    x2image_interactive(input_fn, args, sampler, logger, **kwargs)


def x2text_janus_interactive(input_fn, args, sampler, logger=None, **kwargs):
    """
    A wrapper for starting the interactive inference pipeline.
    """
    if logger is None:
        from loguru import logger
    while True:
        inputs = input_fn()
        if inputs is None:
            break

        # Determine the seed
        if args.seed_type in ["auto", "fixed"]:
            seed = args.seed
        elif args.seed_type == "random":
            seed = None
        else:
            raise ValueError(
                f"When evaluating `prompt`, `seed_type` must be one of ['auto', 'fixed', 'random'], "
                f"got {args.seed_type}."
            )
        # Start sampling
        outputs = sampler.predict(
            **inputs,
            size=args.sample_image_size,
            seed=seed,
            verbose=1,
            **kwargs,
        )
        samples = outputs["samples"]
        logger.info(f"The answer is: {samples}")


def textimage2text_janus_interactive(args, sampler, logger=None, **kwargs):
    def input_fn():
        if args.prompt is None:
            # Ask for the next prompt
            inputs = input("Input prompt (`q` to quit): ")
            if inputs == "q":
                return None
            prompt = inputs
        else:
            prompt = args.prompt
            args.prompt = None
        return {'prompt': prompt}

    x2text_janus_interactive(input_fn, args, sampler, logger, **kwargs)


def label2image_interactive(args, sampler, logger=None, **kwargs):
    def input_fn():
        if args.label is None:
            # Ask for the next prompt
            inputs = input("Input label 0-999 (`q` to quit): ")
            if inputs == "q":
                return None
            label = int(inputs)
        else:
            label = int(args.label)
            args.label = None
        return {'label': label}

    x2image_interactive(input_fn, args, sampler, logger, **kwargs)


def text2image_batch(args, datasource, sampler, logger=None, segments=None, task=None,
                     input_batch_dict=None, **kwargs):
    """
    A wrapper for starting the batch inference pipeline.
    """
    if logger is None:
        from loguru import logger

    save_base = sampler.get_sample_save_dir(
        task=task, testset=Path(args.csv).stem, image_size=args.sample_image_size, segments=segments, rerank=args.rerank,
    )
    logger.info(f"Save the generated images to: {save_base}")
    save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))

    dataloader = sampler.build_sample_dataloader(
        data_source=datasource,
        save_template=save_template,
        batch_size=args.sample_batch_size,
        seed_type=args.seed_type,
        seed=args.seed,
        skip_exist=args.skip_exist,
    )
    sampler.batch_sample(
        dataloader=dataloader,
        input_batch_dict=input_batch_dict,
        rerank=args.rerank,
        # other kwargs passed to predict
        size=args.sample_image_size,
        **kwargs,
    )
    # Print again at final for easy reading
    logger.info(f"Save the generated images to: {save_base}")


def text2siglip_batch(args, datasource, sampler, logger=None, segments=None, task=None,
                     input_batch_dict=None, **kwargs):
    """
    A wrapper for starting the batch inference pipeline.
    """
    if logger is None:
        from loguru import logger

    save_base = sampler.get_sample_save_dir(
        task=task, testset=Path(args.csv).stem, image_size=args.sample_image_size, segments=segments,
    )
    logger.info(f"Save the generated images to: {save_base}")
    save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}" + ".{{}}"))

    dataloader = sampler.build_sample_dataloader(
        data_source=datasource,
        save_template=save_template,
        batch_size=args.sample_batch_size,
        seed_type=args.seed_type,
        seed=args.seed,
        skip_exist=args.skip_exist,
    )
    sampler.batch_sample_siglip(
        dataloader=dataloader,
        input_batch_dict=input_batch_dict,
        rerank=args.rerank,
        # other kwargs passed to predict
        size=args.sample_image_size,
        **kwargs,
    )
    # Print again at final for easy reading
    logger.info(f"Save the generated images to: {save_base}")


def label2image_batch(args, testset, sampler, logger=None, **kwargs):
    """
    A wrapper for starting the batch inference pipeline.
    """
    if logger is None:
        from loguru import logger
    save_base = sampler.get_sample_save_dir(
        testset=f"{testset}_cls{args.n_class}", image_size=args.sample_image_size, rerank=args.rerank,
    )
    logger.info(f"Save the generated images to: {save_base}")
    save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
    dataset = LabelDataset(labels=list(range(args.n_class)),
                           save_template=save_template)
    dataloader = sampler.build_sample_dataloader(
        data_source=dataset,
        batch_size=args.sample_batch_size,
    )
    sampler.batch_sample(
        dataloader=dataloader,
        rerank=args.rerank,
        # other kwargs passed to predict
        size=args.sample_image_size,
        **kwargs,
    )
    # Print again at final for easy reading
    logger.info(f"Save the generated images to: {save_base}")


def x2x_interactive(input_fn, args, sampler, logger=None, processor="image_processor", **kwargs):

    while True:
        inputs = input_fn()
        if inputs is None:
            break

        # Load the image
        if "image_path" in inputs:
            if (image_path := Path(inputs["image_path"])).exists():
                image = Image.open(image_path).convert("RGB")
                inputs['pixel_values'] = sampler.model_dict[processor].preprocess([image]).pixel_values
            else:
                print(f"Invalid image path: {image_path}")
                continue

        # Determine the seed
        if args.seed_type in ["auto", "fixed"]:
            inputs['seed'] = args.seed
        elif args.seed_type == "random":
            inputs['seed'] = None
        else:
            raise ValueError(
                f"When evaluating `prompt`, `seed_type` must be one of ['auto', 'fixed', 'random'], "
                f"got {args.seed_type}."
            )

        # Start generating
        for response in sampler.generate(**inputs, **kwargs):
            if response['type'] == 'text':
                print(response['value'], end='', flush=True)
            elif response['type'] == 'image':
                print(f"[Save the generated image to: {response['save_paths'][0]}]", end='', flush=True)
            else:
                raise ValueError(f"Invalid message type: {response['type']}")
        print()


def image2x_interactive(args, sampler, logger=None, **kwargs):
    def input_fn():
        if args.image is None:
            # Ask for the next image path
            inputs = input("Input image path (`q` to quit): ")
            if inputs == "q":
                return None
            image = inputs
        else:
            image = args.image
            args.image = None

        if args.prompt is None:
            # Ask for the question
            inputs = input("Input question (`q` to quit): ")
            if inputs == "q":
                return None
            prompt = inputs
        else:
            prompt = args.prompt
            args.prompt = None

        return {'image_path': image, 'prompts': [prompt]}

    x2x_interactive(input_fn, args, sampler, logger, **kwargs)


def lm_interactive(args, sampler, logger=None, **kwargs):
    def input_fn():
        if args.question is None:
            # Ask for the question
            inputs = input("[Input question (`q` to quit):] ")
            if inputs == "q":
                return None
            question = inputs
        else:
            question = args.question
            args.question = None

        return {'prompts': [question]}

    x2x_interactive(input_fn, args, sampler, logger, **kwargs)
