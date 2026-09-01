from ..core.global_vars import get_args
from ..core.extra_model_provider import build_text_encoder, build_audio_vae
from ..data_kits.csv_dataset import MessageListDataset
from .hunyuan_multimodal_sampler import HunyuanMultimodalSampler
from ..utils.validation_loss_utils import build_validation_caption_processor
from hymm.data_kits.seeded_dataset import _seeded_rng_ctx

class Leo2Sampler(HunyuanMultimodalSampler):

    def setup_models_from_trainer(self, trainer):
        from hymm.core.global_vars import get_text_encoder, get_audio_vae

        super().setup_models_from_trainer(trainer)
        args = get_args()

        self.model.model_dict["text_encoder"] = get_text_encoder()
        if args.use_audio_vae:
            self.model.model_dict["audio_vae"] = get_audio_vae()

    def setup_extra_models(self):
        super().setup_extra_models()
        args = get_args()

        # Initialize text_encoder
        self.model.model_dict["text_encoder"] = build_text_encoder()
        if args.use_audio_vae:
            self.model.model_dict["audio_vae"] = build_audio_vae(dp_rank=self.rank)

    def prepare_media_config(self, run_task_kwargs, generate_task_specific_kwargs, **kwargs):
        """
        Define some runtime generation configs passed to `LeoModelHF.generate_image()`
        or `LeoModelHF.generate_video()` method.

        `run_task_kwargs` is the `testset_task_kwargs` in hunyuan_multimodal_sampler.py
        """
        args = get_args()
        runtime_config = {**generate_task_specific_kwargs}

        bot_task = run_task_kwargs.get("bot_task", self.model.generation_config.bot_task)
        runtime_config["bot_task"] = bot_task
        media_size = run_task_kwargs.get("image_size", args.image_size)
        if bot_task == "image":
            runtime_config.update(dict(image_size=media_size))
        elif bot_task in ["video", "av"]:
            runtime_config.update(dict(
                video_size=media_size,
                num_frames=run_task_kwargs.get("num_frames", args.num_frames),
                video_fps=run_task_kwargs.get("video_fps", args.video_fps),
                ref_mode=run_task_kwargs.get(
                    "ref_mode",
                    args.ref_mode or getattr(self.model.generation_config, "ref_mode", "sequence"),
                ),
                cond_after_gen=self.parse_bool_kwarg(
                    run_task_kwargs.get("cond_after_gen"),
                    default=self._cond_after_gen_from_train_kwargs(args),
                ),
            ))

        return runtime_config

    @staticmethod
    def _cond_after_gen_from_train_kwargs(args) -> bool:
        for name, value in vars(args).items():
            # task_kwargs may be a plain dict or a mapping-like config (e.g. OmegaConf DictConfig).
            if name.endswith("_task_kwargs") and hasattr(value, "get") \
                    and not isinstance(value, str) and value.get("cond_after_gen", False):
                return True
        return False

    @staticmethod
    def per_batch_config(batch, runtime_config, use_default_image_size=False):
        if runtime_config["bot_task"] in ["image", "video", "av"]:
            # Any-resolution: use per-sample height/width from batch when available.
            media_size_key = "image_size" if "image_size" in runtime_config else "video_size"
            if "height" in batch and "width" in batch:
                media_size = int(batch["height"][0]), int(batch["width"][0])
            else:
                media_size = runtime_config[media_size_key]
            per_batch_runtime_config = {**runtime_config, media_size_key: media_size}

            if "num_frames" in runtime_config and "num_frames" in batch:
                per_batch_runtime_config["num_frames"] = int(batch["num_frames"][0])
            if "video_fps" in runtime_config and "fps" in batch:
                per_batch_runtime_config["video_fps"] = int(batch["fps"][0])

        elif runtime_config["bot_task"] == "audio":
            if "audio_duration" in batch:
                audio_duration = float(batch["audio_duration"][0])
            else:
                audio_duration = runtime_config["audio_duration"]
            per_batch_runtime_config = {**runtime_config, "audio_duration": audio_duration}

        else:
            raise NotImplementedError(f"Unsupported bot_task {runtime_config['bot_task']} in per_batch_config")

        return per_batch_runtime_config

    def build_dataset(self, testset, sample_save_base):
        args = get_args()

        def prompt_fn(prompt, row):
            if args.prompt_prepend_fps:
                fps = row['fps'] if 'fps' in row else args.video_fps
                prompt = f"FPS:{fps}, " + prompt
            if args.prompt_prepend_content:
                prompt = args.prompt_prepend_content + prompt
            if args.prompt_append_content:
                prompt = prompt + args.prompt_append_content
            return {"role": "user", "content": prompt}

        return MessageListDataset(
            testset,
            sample_save_base,
            tokenizer=self.model.tokenizer,
            prompt_fn=prompt_fn,
            skip_existed=args.skip_existed
        )

    def build_validation_dataset(self, testset, sample_save_base, caption_lang="zh",
                                 warn_default_lang=False):
        args = get_args()

        caption_processor = build_validation_caption_processor(args, logger=self.logger)
        if warn_default_lang:
            self.logger.warning(
                f"[validation_loss] neither prompt_zh nor prompt_en given for {testset}; "
                f"defaulting caption language to 'zh'."
            )

        def prompt_fn(prompt, row):
            # Seed the caption augmentation with the sample's own seed (from the CSV `seed`
            # column) so the validation prompt is deterministic and reproducible across ckpts.
            seed = int(row["seed"]) if "seed" in row else 42
            if caption_processor is not None:
                with _seeded_rng_ctx(seed):
                    prompt = caption_processor.caption_aug(prompt, caption_lang)
            if args.prompt_prepend_fps:
                fps = row['fps'] if 'fps' in row else args.video_fps
                prompt = f"FPS:{fps}, " + prompt
            if args.prompt_prepend_content:
                prompt = args.prompt_prepend_content + prompt
            if args.prompt_append_content:
                prompt = prompt + args.prompt_append_content
            return {"role": "user", "content": prompt}

        return MessageListDataset(
            testset,
            sample_save_base,
            tokenizer=self.model.tokenizer,
            prompt_fn=prompt_fn,
            skip_existed=False,
            id_col="fine_md5",
            default_seed=42,
        )