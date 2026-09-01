import loguru
import inspect
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type, Union, List

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers.generation.utils import GenerateOutput

from hymm.core.extra_model_provider import build_vae, build_tkwrapper
from hymm.samplers.entry import create_sampler_for_pipeline
from hymm.core.global_vars import get_args, get_logger, get_parallel_state
from hymm.core.parallel_states import ParallelState
from hymm.data_kits.csv_dataset import MessageListDataset
from hymm.data_kits.datasampler import DistributedSamplerFix
from hymm.metrics import load_metric
from hymm.models import build_model
from hymm.utils.helpers import default, print_args, readable_time
from hymm.utils.file_utils import safe_save_file, save_to_json
from hymm.utils.rank_log import set_rank0_only
from hymm.utils.torch_utils import Timer
from processors.video_kits import assert_saver_available


class HunyuanMultimodalSampler(object):
    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: int = 0,
        logger=None,
        extra_args: Optional[List[str]] = None,
    ):
        """Load sampler for pipeline/inference without distributed.

        Reuses hymm.samplers.entry parsing (parse_argv_from_yaml + add_core_args) so that
        Gradio and run_sample.sh share the same config/arg path.
        extra_args: optional CLI args (e.g. ["--framework", "hf", "--sequence-template", "pretrain"]).
        """
        ckpt_path = Path(ckpt_path)
        config_path = Path(config_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}.")
        if logger is None:
            from loguru import logger as _logger
            logger = _logger
        return create_sampler_for_pipeline(
            config_path=str(config_path),
            ckpt_path=str(ckpt_path),
            extra_args=extra_args,
            sampler_name="hunyuan_multimodal_sampler.HunyuanMultimodalSampler",
            framework="hf",
            device=device,
            logger_instance=logger,
        )

    def __init__(self, init_args, rank, world_size, trainer=None):
        super().__init__()


        # 临时 hardcode 配置检查，确保大家不会用到慢的推理配置
        # TODO: 大家把推理配置更新过来之后，去掉这段检查，保证sampler的通用性
        args = get_args()
        assert args.fsdp_impl == 'new', "Set fsdp_impl to 'new' for sampling to accelerate inference."
        assert 'flash3' in args.attn_impl, "Set attn_impl to 'flash3' for sampling to accelerate inference."
        assert args.gate_impl == 'deepseek', "Set gate_impl to 'deepseek' for sampling to accelerate inference."
        # assert args.moe_impl == 'ep_moe', "Set moe_impl to 'ep_moe' for sampling to accelerate inference."
        if getattr(args, "dp_shard", 1) > 8:
            loguru.logger.warning("Set dp_shard to 8 for sampling to accelerate inference.")
        if getattr(args, "expert_model_parallel_size", 1) > 1:
            loguru.logger.warning("Set expert_model_parallel_size to 1 for sampling to accelerate inference.")
        if getattr(args, "context_parallel_size", 1) > 8:
            loguru.logger.warning("Set cp=8 for higher throughput, cp>8 for lower latency (but lower throughput).")


        self.init_args = init_args
        self.device = rank % torch.cuda.device_count()
        self.rank = rank
        self.world_size = world_size
        self.parallel_state: ParallelState = get_parallel_state()

        set_rank0_only(getattr(init_args, "log_rank0_only", False))

        self.logger = get_logger()
        if trainer is None:
            self.setup_models(init_args.framework)
            self.setup_extra_models()
        else:
            self.setup_models_from_trainer(trainer)
        self.pure_text_tasks = ["auto", "quickly_think", "slowly_think"]
        self.interleaved_tasks = ["interleaved"]

    def setup_models(self, framework):
        args = get_args()

        # Initialize model
        dtype = torch.bfloat16 if args.bf16 and not args.main_params_fp32 else torch.float32
        self.model, self.model_config = build_model(
            args,
            dtype=dtype,
            device=args.init_device,
            initialize_weights=False,
        )
        # Print model config
        if self.rank == 0:
            if isinstance(self.model_config, dict):
                # For MoT
                for key, config in self.model_config.items():
                    if config is not None:
                        print_args(f"{key} model config", config)
            else:
                print_args("model config", self.model_config)

        # Load pretrained weights
        assert args.ckpt is not None, "Checkpoint path `--ckpt` must be provided for sampling."
        if framework == "hf":
            self.model.load_pretrained_model(
                dtype=dtype,
                ckpt_path=args.ckpt,
            )
            self.model.requires_grad_(False)
            self.model.eval()

        elif framework == "fsdp":
            from hymm.engines.hy_base_engine import HyBaseEngine
            from hymm.engines import find_engine

            if not args.infer_skip_load_ckpt:
                self.model.collect_load_plans(
                    load_dir=args.ckpt,
                    fuse_experts_in_load=args.fuse_experts_in_load and args.moe_impl == "ep_moe",
                )
                self.model.load_before_fsdp()

            # Build Model Engine
            ParallelEngine: Type[HyBaseEngine] = find_engine(args.model_name)  # noqa
            self.logger.info("Start building ParallelEngine...")
            self.model_engine: HyBaseEngine = ParallelEngine(
                model=self.model,
                enable_autocast=True,
                autocast_prec=args.autocast_dtype,
                # Put meta param materialization into init_param_and_apply_fsdp2 for memory-efficient initialization
                # For streaming fsdp implementation, we set initialize_meta_param to false, since meta params will be materialized in apply_fsdp.
                initialize_meta_param=args.fsdp_impl == 'new',
                dp_replicate_param_handler='none',
                enable_compile=args.compile_engine,
            )
            self.model.requires_grad_(False)
            self.model_engine.eval()

            # Load from pretrained checkpoint
            if not args.infer_skip_load_ckpt:
                for plan in self.model.after_fsdp_plans:
                    if plan.source == "dcp":
                        default_states = self.model_engine.pre_load_state_dict()
                        self.model_engine.load_checkpoint(**plan.metadata)
                        self.model_engine.post_load_state_dict(default_states)

        else:
            raise NotImplementedError(f"Framework {framework} not supported.")

        self.model.load_generation_config(default(args.generation_config, args.ckpt))

        # Print model structure
        if self.rank == 0:
            print(self.model)

    def setup_models_from_trainer(self, trainer):
        from hymm.core.global_vars import get_vae, get_tkwrapper

        args = get_args()

        self.model = trainer.model
        self.model_config = trainer.model_config
        self.model_engine = trainer.model_engine
        assert args.generation_config is not None, \
            "Generation config must be provided for sampling when loading from trainer."
        self.model.load_generation_config(args.generation_config)

        self.model.tokenizer = get_tkwrapper()
        if args.use_vae:
            self.model.model_dict["vae"] = get_vae()

    def setup_extra_models(self):
        args = get_args()
        # Initialize vae, tokenizer
        self.model.tokenizer = build_tkwrapper()
        if args.use_vae:
            self.model.model_dict["vae"] = build_vae(dp_rank=self.rank, only_encoder=False)

    def run(self):
        args = get_args()
        dp_rank = self.parallel_state.dp_rank
        assert args.prompt is not None, "'prompt' must be provided for generation."
        bot_task = self.model.generation_config.bot_task

        if bot_task in self.pure_text_tasks:
            # Pure text generation
            inputs = self.model.prepare_model_inputs(prompt=args.prompt, image=args.image, bot_task=bot_task)
            self.model.generate(**inputs, verbose=2)

        elif bot_task in self.interleaved_tasks:
            # Interleaved multi-image generation
            generation_outputs = self.model.generate_interleaved(
                prompt=args.prompt[self.rank % len(args.prompt)],
                image=args.image,
                seed=args.seed,
                image_size=args.image_size,
                bot_task=bot_task,
                max_images=args.max_new_images,
                verbose=2,
            )
            texts, images = generation_outputs.texts, generation_outputs.images
            if texts is not None:
                for i, text in enumerate(texts):
                    print(f"[rank {dp_rank}] Text segment {i}: {text}")
            if images is not None:
                for i, image in enumerate(images):
                    image.save(f"image_{dp_rank}_{i}.png")
                    print(f"Image saved to image_{dp_rank}_{i}.png")

        else:
            # Hybrid text-image generation
            generation_outputs = self.model.generate_image(
                prompt=args.prompt[self.rank % len(args.prompt)],
                image=args.image,
                seed=args.seed,
                image_size=args.image_size,
                bot_task=bot_task,
                verbose=2,
            )
            texts, images = generation_outputs.texts, generation_outputs.images
            if texts is not None:
                print(f"[rank {dp_rank}] Generated Text: {texts}")
            for i, image in enumerate(images):
                image.save(f"image_{dp_rank}_{i}.png")
                print(f"Image saved to image_{dp_rank}_{i}.png")

    @staticmethod
    def postprocess_results(save_base):
        results_dir = save_base / "results"
        if not results_dir.exists():
            return

        if torch.distributed.get_rank() == 0:
            all_dfs = []
            for csv_file in sorted(results_dir.glob("results_*.csv"), key=lambda x: int(x.stem.split("_")[-1])):
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
            merged_df = pd.concat(all_dfs, ignore_index=True)
            if 'index' in merged_df.columns:
                # Be tolerant of mixed int/str dtypes in 'index'. This happens
                # when some rank's results_*.csv contains a row whose 'index'
                # was written as a non-numeric string (e.g. inherited from a
                # previously-crashed run via --skip-existed) while other ranks
                # wrote ints. Plain sort_values then raises
                #   TypeError: '<' not supported between instances of 'str' and 'int'
                # crashing rank 0 mid-collective and triggering NCCL timeouts
                # on every other rank. Sort numerically when we can, fall back
                # to string sort otherwise.
                idx_numeric = pd.to_numeric(merged_df['index'], errors='coerce')
                if idx_numeric.notna().all():
                    merged_df['index'] = idx_numeric.astype('int64')
                else:
                    merged_df['index'] = merged_df['index'].astype(str)
                merged_df = merged_df.sort_values(by='index', kind='mergesort')
            merged_save_path = save_base / "results/all_results.csv"
            merged_df.to_csv(merged_save_path, index=False)

    @staticmethod
    def parse_bool_kwarg(value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("0", "false", "no", "off", "n"):
            return False
        if s in ("1", "true", "yes", "on", "y"):
            return True
        return default

    @staticmethod
    def resolve_use_interleaved_dummy(testset_task_kwargs: dict, args) -> bool:
        use_dummy = not getattr(args, "no_interleaved_dummy", False)
        if "use_interleaved_dummy" in testset_task_kwargs:
            use_dummy = HunyuanMultimodalSampler.parse_bool_kwarg(
                testset_task_kwargs["use_interleaved_dummy"], default=use_dummy,
            )
        return use_dummy

    @staticmethod
    def prepare_media_config(testset_task_kwargs, generate_task_specific_kwargs):
        args = get_args()
        config = {**generate_task_specific_kwargs, "image_size": testset_task_kwargs.get("image_size", args.image_size)}
        return config

    @staticmethod
    def per_batch_config(batch, generate_task_specific_kwargs):
        args = get_args()
        use_default_image_size = args.use_default_image_size
        # Any-resolution: use per-sample height/width from batch when available,
        # unless use-default-image-size is set in the inference yaml.
        if not use_default_image_size and "height" in batch and "width" in batch:
            image_size = [(int(h), int(w)) for h, w in zip(batch["height"], batch["width"])]
        else:
            image_size = generate_task_specific_kwargs["image_size"]
        config = {**generate_task_specific_kwargs, "image_size": image_size}
        return config

    def retrieve_collate_field_first(
            self, batch: dict[str, Any], key: str, *, as_int: bool = False,
    ) -> Any:
        v = batch.get(key)
        v = v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else None
        if not as_int:
            return v
        if not isinstance(v, (int, float)):
            return None
        return int(v)

    def _save_interleaved_outputs(self, outputs, batch: dict[str, Any], save_base: Path) -> None:
        """Save interleaved-generation outputs (single batch item -> N images + optional texts).

        Avoids `MultimodalGenerationOutputs.postprocess_outputs` / `save_to` because they assume
        one media element per batch item, which doesn't hold for interleaved multi-image cases.
        """
        is_dummy = batch.get("is_dummy")
        if is_dummy is not None and bool(torch.as_tensor(is_dummy[0]).item()):
            return
        sample_idx = batch["index"][0]
        sample_idx = int(sample_idx.item()) if isinstance(sample_idx, torch.Tensor) else int(sample_idx)

        image_dir = Path(save_base) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_paths: list[str] = []
        pil_frames: list[Any] = []
        if outputs.images is not None:
            for img_idx, image in enumerate(outputs.images):
                image_path = image_dir / f"{sample_idx}_{img_idx}.png"
                image.save(image_path)
                image_paths.append(str(image_path))
                pil_frames.append(image)

        gif_path_str = ""
        if len(pil_frames) > 1:
            try:
                from PIL import Image as PILImage

                gif_path = image_dir / f"{sample_idx}.gif"
                tw, th = pil_frames[0].size
                frames: list[Any] = []
                for im in pil_frames:
                    pil = im.convert("RGB") if getattr(im, "mode", None) != "RGB" else im
                    if pil.size != (tw, th):
                        resample = getattr(
                            getattr(PILImage, "Resampling", PILImage), "LANCZOS", PILImage.LANCZOS,
                        )
                        pil = pil.resize((tw, th), resample)
                    frames.append(pil)
                frames[0].save(
                    gif_path,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    duration=400,
                    loop=0,
                )
                gif_path_str = str(gif_path.resolve())
            except Exception as e:
                self.logger.warning(f"[interleaved] sample {sample_idx}: GIF 保存失败，已跳过 ({e})")

        texts: list[str] = list(outputs.texts) if outputs.texts is not None else []

        results_path = Path(save_base) / "results" / f"results_{self.parallel_state.dp_rank}.csv"
        row = {
            "index": sample_idx,
            "num_images": len(image_paths),
            "gen_images": json.dumps(image_paths, ensure_ascii=False),
            "gen_gif": gif_path_str,
            "gen_texts": json.dumps(texts, ensure_ascii=False),
        }
        # Carry through any scalar batch fields useful for inspection (prompt, seed, ...).
        for key in ("prompt", "seed"):
            val = batch.get(key)
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            if isinstance(val, torch.Tensor):
                val = val.item() if val.ndim == 0 else val.tolist()
            if val is not None:
                row.setdefault(key, val)
        df = pd.DataFrame([row])
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, mode="a", header=not results_path.exists(), index=False)

    def build_dataset(self, testset, sample_save_base):
        args = get_args()
        return MessageListDataset(testset, sample_save_base, tokenizer=self.model.tokenizer, skip_existed=args.skip_existed)

    def run_testsets(self, sample_save_base=None):
        """ args.testsets or args.eval_metrics must be provided.
        'testset' formats definition:
          <dataset>[@@<param1>=<value1>[@@<param2>=<value2>...]]
          param:
            - bot_task
            - max_new_tokens
            - image_size
            - metric: metric type for eval_metric task, e.g., fid, clip_score, fid+clip_score, ...
            - use_interleaved_dummy: when false, ignore dummy_interleaved_* in batch and use normal BOI path
        """
        args = get_args()
        assert args.testsets is not None or args.eval_metrics is not None, \
            "'testsets' or 'eval_metrics' must be provided for run_testsets."
        sample_save_base = default(sample_save_base, args.sample_save_base)
        assert sample_save_base is not None, "'sample_save_base' must be provided for run_testsets."

        run_tasks = [("sample", testset) for testset in (args.testsets or [])] + [
            ("eval_metric", testset) for testset in (args.eval_metrics or [])
        ]

        timer = Timer(enabled=True)
        timer.start("global")
        for task_idx, (run_task_type, testset) in enumerate(run_tasks):
            generate_task_specific_kwargs = dict()

            # 1. Build dataset and dataloader
            dataset = self.build_dataset(testset, sample_save_base)
            testset_task_kwargs = dataset.task_kwargs
            sampler = DistributedSamplerFix(dataset, num_replicas=self.parallel_state.dp_size,
                                            rank=self.parallel_state.dp_rank, shuffle=False, drop_last=False,
                                            add_extra_samples="extend")
            dataloader = DataLoader(dataset, batch_size=args.sample_batch_size, shuffle=False, sampler=sampler,
                                    drop_last=False, collate_fn=getattr(dataset, "collate_fn", None))
            save_base = dataset.save_dir
            self.logger.info(f"=" * 80)
            self.logger.info(f"Running task {testset}({task_idx + 1} / {len(run_tasks)})")
            self.logger.info(f"Save directory: {save_base}")
            self.logger.info(f"=" * 80)

            # 3. Build metric instance if needed
            metric_instances = []
            if run_task_type == "eval_metric":
                kwargs = {}
                if dataset.testset == "mmlu_bench":
                    kwargs = {"tokenizer": self.model.tokenizer}
                    generate_task_specific_kwargs["return_dict_in_generate"] = True
                    generate_task_specific_kwargs["output_logits"] = True
                assert "metric" in testset_task_kwargs, f"'metric' must be specified in testset for eval_metric task."
                metric_types = testset_task_kwargs["metric"].split("+")

                for metric_type in metric_types:
                    metric = load_metric(f"{metric_type}@{dataset.testset}", **kwargs)
                    metric.load_model(self.logger)
                    metric_instances.append((metric, metric_type))

            # 4. Generate and save samples
            bot_task = testset_task_kwargs.get("bot_task", self.model.generation_config.bot_task)
            assert_saver_available(bot_task)
            generate_task_specific_kwargs["bot_task"] = bot_task

            # streamer doesn't support batching output (i.e., verbose == 2).
            verbose = args.verbose if args.sample_batch_size == 1 else min(1, args.verbose)
            if bot_task in self.pure_text_tasks:
                # Pure text generation
                max_new_tokens = int(testset_task_kwargs.get("max_new_tokens", self.model.generation_config.max_new_tokens))
                generate_task_specific_kwargs["max_new_tokens"] = max_new_tokens
                model_type = testset_task_kwargs.get("model_type", None)
                skip_special_tokens = False if model_type else True

                timer.start(f"Task {task_idx}")
                timer.start(f"Batch")
                for batch_idx, batch in enumerate(dataloader):
                    if args.max_sample_batches > 0 and batch_idx >= args.max_sample_batches:
                        break
                    batch: dict[str, Any]
                    self.logger.info(f"Generating batch {batch_idx + 1} / {len(dataloader)} ...")

                    # Inference
                    inputs = self.model.prepare_model_inputs(
                        message_list=batch[dataset.name_mapper("message_list")],
                        mode="gen_text",
                        **generate_task_specific_kwargs,
                    )
                    outputs = self.model.generate(
                        **inputs, decode_text=True, verbose=verbose, skip_special_tokens=skip_special_tokens
                    )
                    outputs = outputs.postprocess_outputs(batch)

                    # Metric processing if needed
                    if run_task_type == "eval_metric" and not outputs.is_empty() and outputs.texts is not None:
                        start_time = time.time()
                        for metric, _ in metric_instances:
                            proc_inputs = {k: v for k, v in batch.items()}
                            proc_inputs["answers"] = outputs.texts
                            metric.process(**proc_inputs, model_type=model_type)
                        # Unwrap results to list of strings
                        if isinstance(outputs.texts, GenerateOutput):
                            outputs.texts = outputs.texts.sequences
                        gen_time = time.time() - start_time
                        self.logger.info(f"Metric process time: {gen_time}")

                    if get_parallel_state().backend_state.is_log_rank(dp_rank=-1):
                        outputs.save_to(
                            save_base=save_base,
                            summary_file_name=f"results/results_{self.parallel_state.dp_rank}.csv",
                        )

                    # Log time
                    timer.stop(f"Batch")
                    self.logger.info(f"Task {testset}({task_idx + 1}/{len(run_tasks)})"
                                     f"[{batch_idx + 1} / {len(dataloader)}] "
                                     f"| {readable_time(timer, 'Batch', len(dataloader) - batch_idx - 1)}")
                    timer.start(f"Batch")
                timer.stop("Batch")
                timer.stop(f"Task {task_idx}")
                self.logger.info(f"Save directory: {save_base}")

            elif bot_task in self.interleaved_tasks:
                # Interleaved multi-image generation (batch_size=1 only)
                generate_task_specific_kwargs = self.prepare_media_config(testset_task_kwargs, generate_task_specific_kwargs)
                max_images = int(testset_task_kwargs.get("max_images", 9))
                max_new_tokens_per_round = int(testset_task_kwargs.get("max_new_tokens_per_round", 2048))
                use_interleaved_dummy = self.resolve_use_interleaved_dummy(testset_task_kwargs, args)
                if not use_interleaved_dummy:
                    self.logger.info("Interleaved dummy path disabled; using normal BOI generation.")

                timer.start(f"Task {task_idx}")
                timer.start(f"Batch")
                for batch_idx, batch in enumerate(dataloader):
                    if args.max_sample_batches > 0 and batch_idx >= args.max_sample_batches:
                        break
                    batch: dict[str, Any]
                    self.logger.info(f"Generating batch {batch_idx + 1} / {len(dataloader)} ...")
                    batch_config = self.per_batch_config(batch, generate_task_specific_kwargs)
                    # Dummy interleaved, now only support **batch_size==1**
                    if use_interleaved_dummy:
                        dummy_text_segments = self.retrieve_collate_field_first(
                            batch, "dummy_interleaved_text_segments",
                        )
                        dummy_num_images = self.retrieve_collate_field_first(
                            batch, "dummy_interleaved_num_images", as_int=True,
                        )
                    else:
                        dummy_text_segments = None
                        dummy_num_images = None
                    outputs = self.model.generate_interleaved(
                        message_list=batch[dataset.name_mapper("message_list")],
                        seed=batch["seed"],
                        max_images=max_images,
                        max_new_tokens_per_round=max_new_tokens_per_round,
                        dummy_text_segments=dummy_text_segments,
                        dummy_num_images=dummy_num_images,
                        **batch_config,
                        verbose=verbose,
                    )
                    # Interleaved produces multiple images for a single batch item, so the default
                    # `postprocess_outputs` / `save_to` (which assume one media per batch element)
                    # are not applicable. Save inline.
                    if get_parallel_state().backend_state.is_log_rank(dp_rank=-1):
                        self._save_interleaved_outputs(outputs, batch, save_base)

                    timer.stop(f"Batch")
                    self.logger.info(f"[Task {testset}({task_idx + 1}/{len(run_tasks)})] "
                                     f"[{batch_idx + 1} / {len(dataloader)}] "
                                     f"| {readable_time(timer, 'Batch', len(dataloader) - batch_idx - 1)}")
                    timer.start(f"Batch")
                timer.stop("Batch")
                timer.stop(f"Task {task_idx}")
                self.logger.info(f"Save directory: {save_base}")

            else:
                # For DiT, it is image/video generation.
                # For AR, it is hybrid text-image generation.
                generate_task_specific_kwargs = self.prepare_media_config(testset_task_kwargs, generate_task_specific_kwargs)
                _image_output = dict(sample="pil", eval_metric="pt")
                _video_output = dict(sample="np", eval_metric="pt")
                _audio_output = dict(sample=dict(audio="pt"), eval_metric=dict(audio="pt"))
                _av_output = dict(sample=dict(visual="np", audio="np"), eval_metric=dict(visual="pt", audio="pt"))
                output_type = dict(
                    image=_image_output,
                    video=_video_output,
                    audio=_audio_output,
                    av=_av_output,
                    think_recaption=_image_output,
                    think=_image_output,
                    recaption=_image_output,
                ).get(bot_task, _image_output)[run_task_type]

                timer.start(f"Task {task_idx}")
                timer.start(f"Batch")
                for batch_idx, batch in enumerate(dataloader):
                    if args.max_sample_batches > 0 and batch_idx >= args.max_sample_batches:
                        break
                    batch: dict[str, Any]
                    self.logger.info(f"Generating batch {batch_idx + 1} / {len(dataloader)} ...")
                    batch_config = self.per_batch_config(batch, generate_task_specific_kwargs)
                    print(f"Batch {batch_idx + 1} / {len(dataloader)} config: {batch_config}")
                    # Inference
                    if bot_task in ["video", "av", "audio"]:
                        r2v_kwargs = {}
                        if "cond_video_vae_path" in batch:
                            r2v_kwargs["cond_video_vae_path"] = batch["cond_video_vae_path"]
                        outputs = self.model.generate_video(
                            message_list=batch[dataset.name_mapper("message_list")],
                            seed=batch["seed"],
                            **batch_config,
                            **r2v_kwargs,
                            output_type=output_type,
                            verbose=verbose,
                        )
                    else:
                        outputs = self.model.generate_image(
                            message_list=batch[dataset.name_mapper("message_list")],
                            seed=batch["seed"],
                            **batch_config,
                            image_output_type=output_type,
                            verbose=verbose,
                        )
                    outputs = outputs.postprocess_outputs(batch)

                    # Metric processing if needed
                    if run_task_type == "eval_metric" and not outputs.is_empty():
                        start_time = time.time()
                        for metric, _ in metric_instances:
                            proc_inputs = {k: v for k, v in outputs.batch.items()}
                            proc_inputs["images"] = outputs.images.float()
                            # GenEvalMetric.process() requires ids (e.g. sample index); batch has "index" or "ids"
                            proc_inputs.setdefault("ids", outputs.batch["index"])
                            metric.process(**proc_inputs)
                        gen_time = time.time() - start_time
                        outputs.postprocess_images(
                            args.eval_save_images,
                            image_processor=self.model.diffusion_pipeline.image_processor,
                        )
                        self.logger.info(f"Metric process time: {gen_time}")

                    if get_parallel_state().backend_state.is_log_rank(dp_rank=-1):
                        outputs.save_to(
                            save_base=save_base,
                            summary_file_name=f"results/results_{self.parallel_state.dp_rank}.csv",
                            fps=batch_config.get("video_fps", args.video_fps),
                            sample_rate=getattr(self.model.generation_config, "audio_sample_rate", None),
                        )

                    # Log time
                    timer.stop(f"Batch")
                    self.logger.info(f"[Task {testset}({task_idx + 1}/{len(run_tasks)})] "
                                     f"[{batch_idx + 1} / {len(dataloader)}] "
                                     f"| {readable_time(timer, 'Batch', len(dataloader) - batch_idx - 1)}")
                    timer.start(f"Batch")
                timer.stop("Batch")
                timer.stop(f"Task {task_idx}")
                self.logger.info(f"Save directory: {save_base}")

            # 4.1 Merge all csv results
            torch.distributed.barrier()
            self.postprocess_results(save_base)

            # 5. Evaluation if needed
            if run_task_type == "eval_metric":
                # Barrier to ensure all processes have finished the evaluation and have the same `self.valid_metrics`
                dist.barrier()
                self.logger.info(f"All processes have finished the evaluation. Start all gathering results...")

                # Synchronization metric inputs
                metric_gather_results = [metric.all_gather_results() for metric, _ in metric_instances]
                self.logger.info(f"All gather results for all metrics finished")

                if self.rank == 0:
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    for (metric, metric_type), gathered_results in zip(metric_instances, metric_gather_results):
                        metric_kwargs = {}
                        accept_save_file = 'save_file' in list(inspect.signature(metric.compute_metrics).parameters.keys())
                        if accept_save_file and hasattr(metric, "save_file_template"):
                            metric_kwargs['save_file'] = save_base / f"metric_temp/{metric_type}" \
                                                         / metric.save_file_template.format(timestamp.replace('-', '_'))
                            self.logger.info(f"[{metric_type}] Temp results will be saved to {metric_kwargs['save_file']}")

                        output = metric.compute_metrics(gathered_results, **metric_kwargs)
                        if isinstance(output, tuple):
                            output = {"_": output}

                        results = []
                        for key, (value, count, *extra_outputs) in output.items():
                            suffix = "" if key == "_" else f"_{key}"
                            results.append({
                                "timestamp": timestamp,
                                "metric": f"{metric_type}{suffix}",
                                "testset": dataset.testset,
                                "value": value,
                                "count": count,
                                "runtime_config": {
                                    **self.model.generation_config.to_dict(),
                                    **generate_task_specific_kwargs,
                                },
                            })
                            if len(extra_outputs) > 0:
                                # Metrics like VQAv2 return a dict to save some extra information
                                if 'extra_info_dict' in extra_outputs[0]:
                                    results[-1]["extra_metric_stats"] = extra_outputs[0]['extra_info_dict']

                        self.logger.info(results)

                        accumulated_results = results[:]
                        save_path = save_base / f"metric_results/{metric_type}.json"
                        if save_path.exists():
                            with open(save_path, "r") as f:
                                ori_results = json.load(f)
                            accumulated_results = ori_results + results

                        save_to = safe_save_file(save_path, accumulated_results, save_fn=save_to_json)
                        self.logger.info(f"Evaluation results saved to {save_to}")

                dist.barrier()

        timer.stop("global")
        self.logger.info(f"Total time cost: {readable_time(timer.elapsed('global'))}.")

    def run_validation_loss_sets(self, sample_save_base=None):
        import numpy as np
        from hymm.utils.validation_loss_utils import (
            parse_validation_loss_set, parse_iter_from_name, variant_cache_dir,
            load_done_indices, append_sample_rows, aggregate_validation_iter,
            plot_validation_losses, sanitize_key,
        )

        args = get_args()
        assert args.validation_loss_sets is not None, \
            "'validation_loss_sets' must be provided for run_validation_loss_sets."
        sample_save_base = default(sample_save_base, args.sample_save_base)
        assert sample_save_base is not None, "'sample_save_base' must be provided for run_validation_loss_sets."
        sample_save_base = Path(sample_save_base)

        # N fixed timesteps in (0, 1), e.g. np.linspace(0, 1, 8+2)[1:-1] for the default N=8.
        n_ts = getattr(args, "validation_loss_timesteps", 8)
        timestep_points = np.linspace(0.0, 1.0, n_ts + 2).tolist()[1:-1]

        # Results accumulate across checkpoints under SAMPLE_SAVE_DIR/validation_loss so the
        # plots (x-axis = ckpt iter) can be re-rendered every round. sample_save_base points at
        # the per-iter dir (e.g. .../samples_metrics_test/iter_0000005).
        val_dir = sample_save_base.parent / "validation_loss"
        iter_num = parse_iter_from_name(sample_save_base.name)
        grank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Debug mode: run normal denoising (self.generate) and save the media instead of computing loss.
        debug_gen = getattr(args, "validation_loss_debug_gen", False)

        # Force re-run: recompute everything and overwrite the cache instead of resuming. 
        force_rerun = getattr(args, "validation_loss_force_rerun", False)

        # Expand every set spec into per-language variants (one dataset pass per variant).
        variants = []
        for spec in (args.validation_loss_sets or []):
            variants.extend(parse_validation_loss_set(spec))

        # With context parallelism (cp_size > 1) all CP ranks that share a dp_rank process the SAME samples
        # collectively (the model forward is a CP collective). The resume/skip decision must therefore be
        # IDENTICAL across the CP group, otherwise some ranks skip a cached sample while others enter the
        # forward and its CP all-gather hangs. So key the cache by dp_rank (shared by all CP ranks) and let
        # only cp_rank 0 write it — every CP rank then loads the same done set and skips in lockstep.
        dp_rank = self.parallel_state.dp_rank
        cp_writer = self.parallel_state.cp_rank == 0

        timer = Timer(enabled=True)
        timer.start("global")
        for v_idx, variant in enumerate(variants):
            cache_dir = variant_cache_dir(val_dir, iter_num, variant)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"rank_{dp_rank}.csv"
            if force_rerun:
                # Overwrite: drop this variant's cached done-set and truncate the single cache file
                # (removing one file, never a directory) so results are rewritten from scratch.
                if cp_writer and cache_file.exists():
                    cache_file.unlink()
                done_indices = set()
            else:
                # Resume: skip samples already fully computed (all timesteps) in a prior run.
                done_indices = load_done_indices(cache_file, n_ts=len(timestep_points))

            dataset = self.build_validation_dataset(
                variant["testset"], sample_save_base,
                caption_lang=variant["lang"],
                warn_default_lang=variant.get("warn_default_lang", False),
            )

            # Latent-path column names are configurable per set via @@video_latent_path/@@audio_latent_path.
            video_latent_col = variant.get("video_latent_col", "latent_cos_path")
            audio_latent_col = variant.get("audio_latent_col", "audio_av_clip_vae_leo_v1_0_0_latent_cos_path")

            # Skip a CSV that lacks the latent columns required to compute the loss. Doing this here (before the
            # compute loop / barriers) is deterministic across ranks, so it can't leave some ranks crashing mid-loop
            # (e.g. KeyError on a missing audio-latent column) while others wait forever at the per-variant barrier.
            required_cols = [video_latent_col, "latent_shape"]
            if getattr(args, "bot_task", "av") in ("av", "audio"):
                required_cols.append(audio_latent_col)
            sample0 = dataset.total_input_dict[0] if len(dataset.total_input_dict) else None
            if sample0 is not None:
                missing_cols = [c for c in required_cols if c not in sample0]
                if missing_cols:
                    self.logger.warning(
                        f"[validation_loss] skipping variant plot={variant['plot_name']} "
                        f"csv={variant['csv_stem']} label={variant['label']}: CSV is missing required "
                        f"column(s) {missing_cols} needed for {getattr(args, 'bot_task', 'av')} validation loss."
                    )
                    continue

            sampler = DistributedSamplerFix(dataset, num_replicas=self.parallel_state.dp_size,
                                            rank=self.parallel_state.dp_rank, shuffle=False,
                                            drop_last=False, add_extra_samples="extend")
            dataloader = DataLoader(dataset, batch_size=args.sample_batch_size, shuffle=False,
                                    sampler=sampler, drop_last=False,
                                    collate_fn=getattr(dataset, "collate_fn", None))

            self.logger.info("=" * 80)
            self.logger.info(f"Validation loss variant {v_idx + 1}/{len(variants)}: "
                             f"plot={variant['plot_name']} csv={variant['csv_stem']} label={variant['label']}")
            self.logger.info(f"Cache directory: {cache_dir}")
            self.logger.info("=" * 80)

            mkey = dataset.name_mapper("message_list")

            # Collect this rank's real (non-dummy) samples for the variant, in dataloader order.
            shard_samples = []
            for batch in dataloader:
                for j in range(len(batch["index"])):
                    if "is_dummy" in batch and bool(batch["is_dummy"][j]):
                        continue  # padding sample added by DistributedSamplerFix("extend")
                    raw_index = batch["index"][j]
                    index = str(raw_index.item()) if torch.is_tensor(raw_index) else str(raw_index)
                    shard_samples.append(dict(
                        index=index,
                        message_list=batch[mkey][j],
                        seed=int(batch["seed"][j]),
                        video_latent_path=batch[video_latent_col][j],
                        audio_latent_path=batch[audio_latent_col][j],
                    ))

            # In debug-gen we always (re)generate; otherwise resume by skipping already-cached samples.
            to_compute = shard_samples if debug_gen else [s for s in shard_samples if s["index"] not in done_indices]

            # CRITICAL for FSDP/CP: the model forward is a collective (all-gather) across ranks, so EVERY rank
            # must call it the SAME number of times, else the collective loses participants and deadlocks. We
            # therefore run max_count = max over ranks of the per-rank to-compute count, and ranks that finish
            # early issue throwaway "padding" forwards (results discarded) purely to stay in lockstep.
            local_count = len(to_compute)
            max_count = local_count
            if torch.distributed.is_initialized():
                _t = torch.tensor([local_count], device=torch.cuda.current_device(), dtype=torch.long)
                torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.MAX)
                max_count = int(_t.item())

            if max_count == 0:
                self.logger.info(
                    f"[{variant['plot_name']}/{variant['label']}] all samples already cached, skipping.")
                continue

            # Fallback sample to drive padding forwards on ranks that have nothing (more) to compute.
            fallback = shard_samples[0] if shard_samples else None
            if fallback is None and len(dataset.total_input_dict):
                rec = dataset.total_input_dict[0]
                fallback = dict(
                    index=None,
                    message_list=rec[mkey],
                    seed=int(rec.get("seed", 42)),
                    video_latent_path=rec[video_latent_col],
                    audio_latent_path=rec[audio_latent_col],
                )

            for step in range(max_count):
                timer.start("Batch")
                is_real = step < local_count
                s = to_compute[step] if is_real else fallback

                outputs = self.model.generate_validation_loss(
                    message_list=s["message_list"],
                    seed=s["seed"],
                    vae_info=self.model.video_processor.video_vae_info,
                    audio_vae_info=self.model.audio_processor.audio_vae_info,
                    video_latent_path=s["video_latent_path"],
                    audio_latent_path=s["audio_latent_path"],
                    timestep_points=timestep_points,
                    args=args,
                )

                # Only persist real (non-padding) results; padding forwards are thrown away.
                if is_real:
                    if debug_gen:
                        if cp_writer:
                            variant_key = sanitize_key(
                                f"{variant['plot_name']}__{variant['csv_stem']}__{variant['label']}")
                            debug_dir = val_dir / "debug_gen" / f"iter_{iter_num:07d}" / variant_key
                            single_batch = {"index": [s["index"]], "is_dummy": torch.tensor([False])}
                            outputs = outputs.postprocess_outputs(single_batch)
                            outputs.save_to(
                                save_base=debug_dir,
                                summary_file_name=f"results/results_{dp_rank}.csv",
                                fps=args.video_fps,
                                sample_rate=getattr(self.model.generation_config, "audio_sample_rate", None),
                            )
                    else:
                        if cp_writer:
                            append_sample_rows(cache_file, s["index"], outputs)
                        done_indices.add(s["index"])

                timer.stop("Batch")
                self.logger.info(f"[{variant['plot_name']}/{variant['label']}] "
                                 f"step {step + 1}/{max_count}"
                                 f"{'' if is_real else ' (padding)'} "
                                 f"| {readable_time(timer, 'Batch', max_count - step - 1)}")

        # Safest design: every rank runs ALL variants fully independently (no per-variant sync at all),
        # so a slow/uneven rank can never stall the others mid-way. Only after the whole compute phase do
        # we sync once and let rank 0 aggregate + plot everything in a single shot. The compute loop above
        # needs no barriers: CP correctness comes from the shared per-dp_rank cache (identical skip
        # decisions across a CP group), not from barriers.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if grank == 0 and not debug_gen:
            # Crash-safe: an aggregation/plot error on rank 0 must NOT prevent it from reaching the final
            # barrier, otherwise every other rank would deadlock waiting for it.
            try:
                self.logger.info("[validation_loss] all variants done; aggregating + plotting ...")
                aggregate_validation_iter(
                    val_dir, iter_num, timestep_points, variants,
                    flow_shifts={
                        "video": getattr(args, "flow_shift_video", None),
                        "audio": getattr(args, "flow_shift_audio", None),
                    },
                )
                plot_validation_losses(val_dir)
                self.logger.info("[validation_loss] aggregate + plot done.")
            except Exception as exc:
                self.logger.warning(f"[validation_loss] aggregate/plot failed (continuing): {exc}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        timer.stop("global")
        self.logger.info(f"Total validation-loss time cost: {readable_time(timer.elapsed('global'))}.")

    @staticmethod
    def safe_destroy_process_group(group=None):
        """
        在当前镜像+当前机器下，调用 ``destroy_process_group`` 可能随机触发段错误。
        该问题在正常使用中可复现，根因仍未明确；可能因素包括 PyTorch 版本、运行镜像或硬件。

        崩溃现场显示 RIP 指针可能跳到非法地址（非代码地址），这通常提示存在内存破坏，
        例如 ``destroy_process_group`` 内部或此前 PyTorch 执行路径中的溢出。
        也不能排除硬件随机故障（如内存不稳定），尤其是实验中经常观察到同一时刻大批量出现段错误。
        在该问题定位并修复前，本方法只等待未完成工作，不执行可能 segfault 的 destroy 操作。
        """
        from torch.distributed import ProcessGroup, GroupMember
        from torch.distributed.distributed_c10d import _world

        if group == GroupMember.NON_GROUP_MEMBER:
            return

        if group is None:
            pg = GroupMember.WORLD
        else:
            pg = group

        assert pg is not None
        if _world.pg_map.get(pg, None) is None:
            raise ValueError("Invalid process group specified")

        if type(pg) == ProcessGroup and pg._has_hooks():
            pg._wait_for_pending_works()



    def exit(self):
        torch.cuda.empty_cache()

        if torch.distributed.is_initialized():
            dist.barrier()
            torch.cuda.synchronize()
            print(f"[rank {self.rank}] Sampling is complete. Exiting now.", flush=True)
            self.safe_destroy_process_group()
