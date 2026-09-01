import argparse
from .constants import *
from .utils.helpers import as_tuple, merge_yaml_files, default
from easydict import EasyDict
from loguru import logger
import yaml


def resolve_values(value):
    """ Resolve sub_value of yaml config file. """
    if isinstance(value, dict):
        return EasyDict(value)
    if isinstance(value, list):
        return [resolve_values(v) for v in value]
    return value


def check_config_key_conflicts(config_dict, keys):
    # We don't allow the same key in different groups
    for key, value in config_dict.items():
        if key in keys:
            raise ValueError(f"Key conflict: {key}")
        keys.add(key)
        if isinstance(value, dict):
            check_config_key_conflicts(value, keys)


def flat_dict(d, with_parent=False):
    flatted = {}
    for key, value in d.items():
        if with_parent:
            flatted[key] = resolve_values(value)
        if isinstance(value, dict):
            flatted.update(flat_dict(value, False))
        elif isinstance(value, (list, tuple)) and all(isinstance(v, dict) for v in value):
            for v in value:
                flatted.update(flat_dict(v, False))
        else:
            flatted[key] = value
    return flatted


def add_ptm_args(parser: argparse.ArgumentParser):
    from .utils.lr_schedules import add_tuning_arguments

    # ========= yaml parser ==========
    defaults = {}
    # config from argparse
    config_parser = argparse.ArgumentParser(description="Inner parser for the config file")
    config_parser.add_argument("--config-path", type=str, help="Path to the yaml config.")
    config_args, remain_args = config_parser.parse_known_args()

    # config from base.yaml
    config_yamls = [BASE_CONFIG_PATH]
    if config_args.config_path is not None:
        config_yamls.append(config_args.config_path)
    # Merge yaml files (contains recursive dict, remove the first level)
    default_yaml_config = merge_yaml_files(config_yamls)
    try:
        check_config_key_conflicts(default_yaml_config, set())
    except ValueError as e:
        print(default_yaml_config)
        raise e
    # Flatten the config dict. (i.e., mirror and lift all the nested key-value pairs to the top level)
    defaults.update(flat_dict(default_yaml_config, with_parent=True))
    
    # ========= extra_args from add_dit_args ==========
    kwargs = dict(ptm=True)
    # parser = add_logging_args(parser, **kwargs)
    parser = add_model_args(parser, **kwargs)
    parser = add_extra_models_args(parser, **kwargs)
    parser = add_denoise_schedule_args(parser, **kwargs)
    parser = add_data_args(parser, **kwargs)
    parser = add_data_yaml_args(parser, **kwargs)
    parser = add_deepspeed_args(parser, **kwargs)
    parser = add_ema_args(parser, **kwargs)
    parser = add_training_args(parser, **kwargs)
    parser = add_evaluation_args(parser, **kwargs)
    parser = add_tools_args(parser, **kwargs)
    parser = add_tuning_arguments(parser, **kwargs)

    group = parser.add_argument_group(title="dit")

    # align yaml
    group.add_argument("--block_size", type=int,
                    help="align model_kwargs.block_size")
    group.add_argument("--use-ptm", action="store_true", help="use ptm in model")
    group.add_argument("--use-final-time-embed", action="store_true", help="use final time embed")
    group.add_argument('--no-init-optim', action='store_true',
                       help='Do not init optimizer, useful when converting checkpoint.')
    # ========= set args from yaml as default ==========
    parser.set_defaults(**defaults)

    return parser


def parse_args(argv=None, defaults=None, namespace=None):
    from .utils.lr_schedules import add_tuning_arguments

    # ----- Config parser -----
    defaults = defaults or {}
    # config from argparse
    config_parser = argparse.ArgumentParser(description="Inner parser for the config file")
    config_parser.add_argument("--config-path", type=str, help="Path to the yaml config.")
    config_args, remain_args = config_parser.parse_known_args(argv)
    # config from base.yaml
    config_yamls = [BASE_CONFIG_PATH]
    if config_args.config_path is not None:
        config_yamls.append(config_args.config_path)
    # Merge yaml files (contains recursive dict, remove the first level)
    default_yaml_config = merge_yaml_files(config_yamls)
    try:
        check_config_key_conflicts(default_yaml_config, set())
    except ValueError as e:
        print(default_yaml_config)
        raise e
    # Flatten the config dict. (i.e., mirror and lift all the nested key-value pairs to the top level)
    defaults.update(flat_dict(default_yaml_config, with_parent=True))

    # ----- Main parser -----
    parser = argparse.ArgumentParser(description="Hunyuan Autoregressive Multimodal training/inference script",
                                     # Disable abbreviations to avoid unexpected behavior
                                     allow_abbrev=False,
                                     )

    # Common
    parser = add_logging_args(parser)
    parser = add_tools_args(parser)
    parser = add_model_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_deepspeed_args(parser)
    parser = add_fsdp_args(parser)
    parser = add_data_args(parser)
    parser = add_denoise_schedule_args(parser)
    # Train
    parser = add_ema_args(parser)
    parser = add_training_args(parser)
    parser = add_tuning_arguments(parser)
    # Sample/Eval
    parser = add_evaluation_args(parser)

    parser.set_defaults(**defaults)
    args = parser.parse_args(args=remain_args, namespace=namespace)

    # ---------------------------------------------------
    # Only modify the args here, don't in other places.
    args = manual_assign(args, remain_args)
    args = sanity_check_args(args)
    # ---------------------------------------------------

    # Convert Namespace to EasyDict for uniform access
    args = EasyDict(vars(args))
    set_launcher(args.launcher)
    return args


def parse_eval_initial_args():
    """
    Note: Arguments in this section are duplicated from other sections for procedure initialization
          before the main parser is called.
    """
    parser = argparse.ArgumentParser("Initial arguments for evaluation.")
    parser.add_argument("--ddp", action="store_true", help="Enable Distributed Data Parallel (DDP) during sampling.")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for DDP.")
    parser.add_argument("--node-index", type=int, default=0, help="Node index for DDP.")
    parser.add_argument("--deepspeed", action="store_true", help="Enable Deepspeed during sampling.")
    parser.add_argument("--tp-size", type=int, default=1, help="TP size.")

    parser.add_argument("--reproduce", action="store_true", help="Enable reproducibility by setting random seeds and deterministic algorithms.")
    parser.add_argument("--global-seed", type=int, default=1, help="Global seed for reproducibility.")

    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--task", type=str, default="sample", help="Inference task.")

    # use ptm inference
    parser.add_argument("--use-ptm", action="store_true", help="use ptm in model")

    # use huggingface inference
    parser.add_argument("--use-hf", action="store_true", help="use huggingface inference")

    # puretorch related
    parser.add_argument("--ep-size", type=int, default=1, help="Size of expert parallel.")
    parser.add_argument("--pp-size", type=int, default=1, help="Size of pipeline parallel.")
    parser.add_argument("--pp-splits", type=str, default=None, help="1,2,3,4这种")
    parser.add_argument("--launcher", type=str, default=1, help="")
    parser.add_argument("--puretorch-ckpt", type=str, default=None, help="")

    args, _ = parser.parse_known_args()

    if args.launcher == 'pure_torch':
        mode = "pure_torch"
    elif args.ddp:
        mode = "ddp"
    elif args.use_ptm:
        mode = "ptm"    
    elif args.deepspeed:
        mode = "deepspeed"
    else:
        mode = "none"
    return args, mode


# ======================================================
#     Define core args groups
# ======================================================

def add_data_core_args(parser, ptm: str | bool = False):
    if ptm != "v1":
        parser.add_argument("--tokenizer-name", type=str, help="Tokenizer name.")
        parser.add_argument("--tokenizer-class", type=str, default="tokenizer_wrapper.TokenizerWrapper",
                            help="Tokenizer class to use.")

    # index-kits
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffle for data loading. (Highest priority: implement in index-utils.py, skips `self.index_manager.shuffle` in `IndexDataset.shuffle`)")
    parser.add_argument("--first-epoch-no-shuffle", action="store_true", help="Disable shuffle for data loading.")
    parser.add_argument("--shuffle-by", type=str, default="dataset", choices=["dataset", "sampler"],
                        help="Shuffle data by dataset or sampler.")
    parser.add_argument("--fast-shuffle", action="store_true", help="Enable fast shuffle for data loading.")
    parser.add_argument("--cache-shuffle", action="store_true", help="Enable cache shuffle for data loading.")
    parser.add_argument("--use-numpy-indices", action="store_true", help="Use numpy indices for index loading.")
    parser.add_argument("--distributed-sampler-low-cpu-memory", action="store_true",
                        help="Use low cpu memory mode for distributed sampler.")

    # media path
    parser.add_argument("--cos-base", type=str, nargs="+",
                        help="Base path for COS storage of images. It should be with the format "
                             "`/cos_nj1/share_xxxx -> /cos_zw1/share_xxxx`.")
    parser.add_argument("--cos-file-is-encrypted", action="store_true", help="Whether the COS files are encrypted.")
    parser.add_argument("--video-cos-base", type=str, help="Prefix for video COS storage.")
    parser.add_argument("--image-cos-base", type=str, help="Prefix for image latent COS storage.")
    parser.add_argument("--last-frame-cos-base", type=str, default=None,
                        help="Prefix for fl2v/fl2va last-frame latent COS storage. "
                             "Falls back to --video-cos-base when unset.")
    parser.add_argument("--audio-cos-base", type=str, help="Prefix for audio COS storage.")

    # vae specific
    parser.add_argument("--vae-image-token-length", type=int, help="Maximum image token length of vae latents.")
    parser.add_argument("--vae-video-token-length", type=int, help="Maximum video token length of vae latents.")
    parser.add_argument("--vae-audio-token-length", type=int, help="Maximum audio token length of vae latents.")
    # -- duration group (for video)
    parser.add_argument("--duration-range", type=int, nargs=2, help="Duration range for video data.")
    parser.add_argument("--duration-step", type=int, default=8, help="Duration step size for video data.")
    parser.add_argument("--additional-durations", type=int, nargs="*", help="Additional durations for video data.")
    # -- resolution group
    parser.add_argument("--reso-strategy", type=str, choices=["reso_group", "anyres"], default="reso_group", help="Resolution strategy.")
    # for reso_group strategy
    parser.add_argument("--reso-base-size", type=int, help="Resolution group base size.")
    parser.add_argument("--reso-align", type=int, default=16, help="Resolution alignment.")
    parser.add_argument("--reso-mode", type=str, help="Resolution mode.")
    parser.add_argument("--reso-preset", type=str, help="Resolution group preset.")
    parser.add_argument("--reso-step", type=int, help="[sdxl mode] Resolution step size.")
    parser.add_argument("--reso-aspect-ratios", type=str, nargs="*",
                        help="[aspect ratio mode] Aspect ratios for resolution group.")
    parser.add_argument("--reso-num-buckets", type=int, help="[arc mode] Number of buckets for resolution group.")
    parser.add_argument("--reso-added-method", type=str, default="append", help="Added method for resolution group.")
    parser.add_argument("--extend-vae-reso-group", action="store_true",
                        help="Extend vae reso group for special ratios of HYImage3.0")
    # for anyres mode
    parser.add_argument("--reso-max-area", type=int, help="Resolution group max area.")
    parser.add_argument("--reso-max-size", type=int, help="Resolution group max size.")
    # reuse --reso-align and --reso-base-size in reso_group mode for anyres mode

    # vit specific
    parser.add_argument("--vit-image-token-length", type=int, help="Image token length of vit features.")
    parser.add_argument("--min-vit-image-token-length", type=int, help="Minimum image token length of vit features.")
    # Image specific
    parser.add_argument("--pixel-resample-type", type=str, default="bicubic",
                        choices=["linear", "bicubic", "lanczos"],
                        help="Resample type for pixel-level resizing.")
    parser.add_argument("--add-iw-ih-token", action="store_true",
                        help="Add iw and ih token before the image token sequence.")
    parser.add_argument("--add-timestep-token", action="store_true", help="Add timestep token.")
    parser.add_argument("--add-timestep-r-token", action="store_true", help="Add timestep r token.")
    parser.add_argument("--add-guidance-token", action="store_true", help="Add guidance token.")
    parser.add_argument("--add-image-shape-token", action="store_true",
                        help="Add image shape token before the image token sequence.")
    parser.add_argument("--add-tw-th-token", action="store_true",
                        help="Add tw/th token (token_width, token_height from tw_th vocab) before the image token sequence.")
    parser.add_argument("--use-front-boi-token", action="store_true",
                        help="Put boi token in the front of iw, ih and timestep tokens.")
    parser.add_argument("--use-front-src-image", action="store_true",
                        help="Put src image in the front of user prompts.")
    parser.add_argument("--ignore-boi-token", action="store_true",
                        help="ignore boi token.")
    parser.add_argument("--ignore-bot-sep-token", action="store_true",
                        help="If True, the bot_sep token (separator at the end of bot turns) will be excluded from"
                             "the training loss by setting `ignore=True` on its sections.")
    parser.add_argument("--use-joint-image-feature", action="store_true",
                        help="Use joint vae and vit image features.")
    parser.add_argument("--gen-image-template", type=str, default="default",
                        choices=["default", "tool_call_v3.5", "dit_it", "multi_stream_dit"],
                        help="[Deprecated] Image template for generation.")
    parser.add_argument("--gen-template", type=str, help="Generation template for all modalities."
                                                         "If specified, it will override the gen-image-template.")

    # Video specific
    parser.add_argument("--add-video-timestep-token", action="store_true", help="Add video timestep token.")
    parser.add_argument("--add-video-guidance-token", action="store_true", help="Add video guidance token.")
    parser.add_argument("--add-video-shape-token", action="store_true", help="Add video shape token.")
    parser.add_argument("--video-data-format", type=str, choices=["video", "latents"], default="latents")
    # image-data-format / last-frame-data-format 改为按 dataset 在 {tag}_task_kwargs 里配置
    # （image_data_format / last_frame_data_format），与 audio_data_format 一致，不再是全局参数。
    parser.add_argument("--gen-video-template", type=str, default="dit_it",
                        choices=["default", "dit_it", "multi_stream_dit"], help="[Deprecated] Video template for generation.")
    parser.add_argument("--vit-video-token-length", type=int,
                        help="Max post-merge vit tokens for a whole condition clip, all frames included.")
    parser.add_argument("--min-vit-video-token-length", type=int, default=128,
                        help="Minimum post-merge vit tokens for a whole condition clip. Clips below this "
                             "budget get upscaled by smart_resize.")

    # Audio specific
    parser.add_argument("--audio-channels", type=int, choices=[1, 2], help="Number of audio channels. (redundant field, pending deprecation)")
    parser.add_argument("--audio-sample-rate", type=int, help="Audio sample rate. (redundant field, pending deprecation)")
    parser.add_argument("--gen-audio-template", type=str, default="default",
                        choices=["default", "dit_it", "multi_stream_dit"], help="[Deprecated] Audio template for generation.")

    # multimodal data
    parser.add_argument("--modality", type=str, nargs="*",
                        choices=["text", "vae_image", "vit_image", "vae_video", "vae_audio"],
                        help="Data modality for training and inference.")
    parser.add_argument("--sequence-template", type=str, default="instruct",
                        choices=["pretrain", "instruct"], help="Choose which template to use.")
    parser.add_argument("--conv-template", type=str, help="Type of pretraining and instruction template. If None, "
                                                          "default to the template the same with model name.")
    parser.add_argument("--sampling-probs", type=str, help="Probability of sampling datasets.")
    parser.add_argument("--combined-iterator-sampling-mode", type=str, default="fixed", choices=["fixed", "random"],
                        help="Combined iterator sampling mode.")
    parser.add_argument("--ignore-start-tokens", type=str, nargs="*", default=[],
                        help="List of tokens to ignore at the start.")
    parser.add_argument("--interleave-only-last-round", action="store_true", help="Only include the last round of generation in the interleave data.")
    parser.add_argument("--dummy-type", type=str, nargs="*", default=[], help="Dummy token type. Can include 'vit' and 'ts_proj'.")

    # Unconditional
    parser.add_argument("--uncond-length", type=str, choices=["single", "equal"], default="equal",
                        help="Unconditional token length for replacing the original prompt.")

    # Sequence pack
    parser.add_argument("--pack-buffer-factor", type=int, default=4, help="A factor to determine the pack buffer size.")
    parser.add_argument("--independent-seq-length", action="store_true",
                        help="If enabled, --max-token-length will be used as the maximum sequence length for "
                             "sequence pack.")
    parser.add_argument("--pack-seq-reduce-pad", action="store_true", help="Whether to reduce pad token for sequence pack.")
    parser.add_argument("--und-pad-min", action="store_true", 
                        help="ensuring the minimum length of und_pad; otherwise, the minimum length of gen_pad is ensured.")
    parser.add_argument("--use-rope-sample-offsets", action="store_true", help="Whether to use sample offsets for rope.")

    # tools
    parser.add_argument("--enable-resample", action="store_true", help="Enable resample for error in data loading.")
    parser.add_argument("--drop-resampled-samples", action="store_true",
                        help="Drop the resampled data. Now DiT only supports stackable data, but resampled data may "
                             "have different image size, which is not stackable. This flag can be used to drop the "
                             "resampled data to avoid errors, with the cost of introducing different micro batch size.")
    if not ptm:
        parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading.")
        parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor for data loading.")

    return parser


def add_model_core_args(parser, ptm: str | bool = False):
    # precision
    if not ptm:
        parser.add_argument("--bf16", action="store_true", help="Use bf16 precision for training the model.")
        parser.add_argument("--main-params-fp32", action="store_true", default=True,
                            help="Use fp32 precision for main parameters (e.g., model weights) during training.")
        parser.add_argument("--main-params-bf16", action="store_false", dest="main_params_fp32",
                            help="Use bf16 precision for main parameters (e.g., model weights) during training.")
    parser.add_argument("--autocast-dtype", type=str, choices=PRECISIONS,
                        help="Autocast precision mode for training the model.")
    parser.add_argument("--fsdp-cast-forward-inputs", action="store_true",
                        help="Whether to cast inputs of forward to the same precision as parameters when using FSDP.")
    parser.add_argument("--fsdp-impl", type=str, default="streaming", choices=["new", "streaming"],
                        help="FSDP implementation to use in engine.")
    # reproducibility and performance
    parser.add_argument("--reproduce", action="store_true",
                        help="Enable reproducibility by setting random seeds and deterministic algorithms.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Enable torch cudnn benchmark mode for potential performance improvement.")
    parser.add_argument("--no-benchmark", action="store_false", dest="benchmark",
                        help="Disable torch cudnn benchmark mode.")
    # model structure
    if not ptm:
        # Basic
        parser.add_argument("--seed", type=int, default=1234, help="Global seed.")
        parser.add_argument("--vocab-size", type=int, help="Vocabulary size.")
        parser.add_argument("--num-layers", type=int, help="Number of layers.")
        parser.add_argument("--hidden-size", type=int, help="Hidden size.")
        parser.add_argument("--seq-length", type=int, help="Data sequence length.")
        parser.add_argument("--max-position-embeddings", type=int, help="Max position embeddings.")
        parser.add_argument("--num-attention-heads", type=int, help="Number of attention heads.")
        parser.add_argument("--ffn-hidden-size", type=int, help="FFN hidden size.")

    parser.add_argument("--hidden-size-mot-gen", type=int, help="Hidden size for MoT generation.")
    parser.add_argument("--moe-topk-mot-gen", type=int, help="MoE topk for MoT generation.")

    # sequence
    parser.add_argument("--cond-image-type", type=str, choices=["vae", "vit", "vae_vit"], default="vae_vit",
                        help="Feature composition of conditional images.")
    
    # reference/edit video generation
    parser.add_argument("--cond-video-type", type=str, choices=["none", "vae", "vit", "vae_vit"], default="none",
                        help="Feature composition of conditional (reference/source) videos for r2v editing. ")
    parser.add_argument("--cond-video-vit-fps", type=float, default=2.0,
                        help="Rate a condition/source video is uniformly sampled at for ViT encoding, so that "
                             "longer clips contribute more frames (used when cond-video-type includes 'vit').")
    parser.add_argument("--cond-video-vit-max-frames", type=int, default=32,
                        help="Upper bound on the frames taken at --cond-video-vit-fps, which caps how much of "
                             "the sequence budget one condition video can claim.")
    parser.add_argument("--cond-video-vit-num-frames", type=int, default=None,
                        help="Pin the number of frames uniformly sampled from a condition/source video for ViT "
                             "encoding, ignoring --cond-video-vit-fps and --cond-video-vit-max-frames.")
    parser.add_argument("--cond-video-frame-cache-interval-sec", type=float, default=None,
                        help="Seconds between the images of a condition-video frame cache. ")
    parser.add_argument("--cond-vae-zero-timestep", action="store_true", default=True,
                        help="enable to use clean vae cond")
    parser.add_argument("--no-cond-vae-zero-timestep", action="store_false", dest="cond_vae_zero_timestep",
                        help="disable clean vae cond")

    # channel
    parser.add_argument("--extend-latent-channels", action="store_true",
                        help="Extend latent channels for i2v and fl2v.")
    parser.add_argument("--img-latent-in-channels", type=int, help="Number of input channels of image/video projector.")

    # attention
    parser.add_argument("--split-qkv", action="store_true", default=None, help="Split QKV matrices.")
    parser.add_argument("--no-split-qkv", action="store_false", dest="split_qkv")
    parser.add_argument("--cond-token-attn-type", type=str,
                        choices=["causal", "full", "joint_full", "full_causal"], default="joint_full",
                        help="What type of attention mask to use for conditional image tokens in llm.")
    parser.add_argument("--timestep-vae-full-attn", action="store_true",
                        help="Include timestep tokens in the same full-attention slice as VAE tokens.")
    parser.add_argument("--cond-video-token-attn-type", type=str, default="none",
                        choices=["none", "causal", "full"],
                        help="What type of attention mask to use for conditional video tokens in llm.")
    parser.add_argument("--attn-impl", type=str, choices=["sdpa", "flex", "flash", "flash3", "flash_packed", "flash3_packed", "sageattn"],
                        help="Attention implementation to use.")
    parser.add_argument("--inference-attn-impl", type=str,
                        choices=["sdpa", "flex", "flash", "flash3", "flash_packed", "flash3_packed", "sageattn"],
                        help="Optional attention implementation used when gradients are disabled.")

    # mlp
    parser.add_argument("--split-gate-and-up", action="store_true", default=None,
                        help="Split gate and up projection matrices in MLP.")
    parser.add_argument("--no-split-gate-and-up", action="store_false", dest="split_gate_and_up")
    parser.add_argument("--text-branch-split-gate-and-up", action="store_true", default=None,
                        help="Split gate and up projection matrices in MLP for text branch.")
    parser.add_argument("--audio-branch-split-gate-and-up", action="store_true", default=None,
                        help="Split gate and up projection matrices in MLP for audio branch.")

    # moe
    parser.add_argument("--moe-impl", type=str, choices=["hunyuan", "deepseek", "flashinfer", "qwen3", "ep_moe"],
                        help="MoE implementation.")
    parser.add_argument(
        "--ep-moe-weight-format", type=str, choices=["ep_moe", "flashinfer"], default="ep_moe",
        help="Expert weight storage for ep_moe fused experts. Only used when --moe-impl=ep_moe. "
             "'ep_moe': separate gate/up/down_proj_weights; "
             "'flashinfer': fused expert_gate_and_up_weights (FlashInfer-compatible names under experts.).",
    )
    parser.add_argument("--moe-drop-tokens", action="store_true", help="Drop tokens for MoE layers.")
    parser.add_argument("--fuse-experts-in-load", action="store_true",
                        help="Whether to fuse experts weights in loading.")
    parser.add_argument("--gate-impl", type=str, choices=["hunyuan", "deepseek", "ep_moe"],
                        help="Gate implementation.")
    if not ptm:
        parser.add_argument("--num-experts", type=int, help="Number of experts.")
        parser.add_argument("--moe-ffn-hidden-size", type=int, help="FFN hidden size for MoE layers.")

    # mot
    parser.add_argument("--use-mot", action="store_true", help="Use MoT.")
    parser.add_argument("--und-token-type", type=str, nargs="*",
                        default=["bos", "text", "boi", "vit", "eoi", "joint_image_sep", "eoi", "eos", "pad"],
                        help="Und token type.")
    parser.add_argument("--gen-token-type", type=str, nargs="*", default=["vae_info", "vae"], help="Gen token type.")
    parser.add_argument("--copy-mot-in-load", action="store_true",
                        help="Whether to copy mot weights in loading.")
    parser.add_argument("--mot-und-frozen", action="store_true",
                        help="Whether to freeze the und stream in MoT, not including the embedding layer, final norm layer and lm head.")
    parser.add_argument("--mot-gen-frozen", action="store_true",
                        help="Whether to freeze the gen stream in MoT.")
    parser.add_argument("--mot-embed-norm-lm-head-frozen", action="store_true",
                        help="Whether to freeze model.embed_tokens, model.norm and lm_head (when using MoT).")
    parser.add_argument("--gen-branch-model-name", type=str, help="Generation branch model name.")

    # mmdit
    parser.add_argument("--text-branch-model-name", type=str, help="Model name for the text branch of mmdit.")
    parser.add_argument("--audio-branch-model-name", type=str, help="Model name for the audio branch of mmdit.")
    parser.add_argument("--audio-token-type", type=str, nargs="*", default=["audio"], help="Audio token type.")
    parser.add_argument("--use-input-ids", action="store_true",
                        help="Use input ids for mmdit (A simpler and unified multi-modal input layer).")
    parser.add_argument("--text-branch-num-layers", type=int, help="Number of layers for the text branch of mmdit.")

    # Basic
    parser.add_argument("--model-name", type=str, help="Model name.")
    parser.add_argument("--model-structure", type=str, help="Model structure.")
    parser.add_argument(
        "--text-states-hidden-dim",
        type=int,
        help="Must match the text encoder last hidden (e.g. 2048 for Qwen3.5-35B-A3B, 4096 for many dense TEs).",
    )
    parser.add_argument('--use-compile', action='store_true', help='Use torch.compile to compile attn blocks online.')
    parser.add_argument('--compile-engine', action='store_true', help='Compile parallel engine')

    parser.add_argument("--key-mapping", type=str, nargs="*",
                        help="Module key mapping for loading checkpoint, specified as pattern:replacement. "
                             "Regex is supported.")
    parser.add_argument("--ignore-unexpected-keys", type=str, nargs="*",
                        help="Prefix of module keys to ignore for unexpected keys.")
    parser.add_argument("--swap-gate-and-up", action="store_true",
                        help="For the weights of gate_and_up_proj layer, torch implements SwiGLU with the first half "
                             "being up_proj and the second half being gate_proj (although we name it 'gate_and_up'), "
                             "while PTMv2 using TEGroupedMLP implements the opposite. Our inference engine(FlashInfer) "
                             "applies the torch version, so we need to swap the two halves when loading PTMv2 weights.")
    parser.add_argument("--load-remove-prefix", action="store_true",
                        help="Remove the 'model.' prefix when loading checkpoint, useful for PTMv2 checkpoints.")

    # Position Embedding
    parser.add_argument("--rope-type-extended", type=str, help="Rope type.")
    parser.add_argument("--rope-scaling", type=float, help="Rope scaling.")
    parser.add_argument("--rope-theta", type=float, help="RoPE theta. Default to model default.")
    parser.add_argument("--rope-dim-list", type=int, nargs="*", help="RoPE dim list for nd-RoPE.")
    parser.add_argument("--rope-audio-rescale-factor", type=float, default=1.0, help="RoPE audio rescale factor.")
    parser.add_argument("--rope-no-space", action="store_true", help="Whether to use no space for RoPE.")
    parser.add_argument("--rope-fixed-space", type=int, default=None,
                        help="Fixed RoPE space length S between modalities. Only effective when "
                             "rope_no_space is False. When set, the media span occupies S positions "
                             "centered at L (last text end) regardless of (d, h, w).")
    parser.add_argument("--rope-cond-image-fixed-space", type=int, default=None,
                        help="Fixed RoPE span S reserved for each conditioning image "
                             "(cond_vit_image / cond_vae_image). Only effective when rope_no_space is False.")
    parser.add_argument("--rope-cond-video-fixed-space", type=int, default=None,
                        help="Fixed RoPE span S reserved for each conditioning video "
                             "(cond_vit_video / cond_vae_video). Only effective when rope_no_space is False.")

    parser.add_argument("--fp32-modules", type=str, nargs="*", help="Modules to use fp32 precision.")

    return parser


def add_extra_models_core_args(parser: argparse.ArgumentParser, ptm: str | bool = False):
    # vae
    parser.add_argument("--use-vae", action="store_true", help="Whether to use VAE model.")
    parser.add_argument("--use-vae-decoder", action="store_true", help="Whether to load VAE decoder. If False, only load VAE encoder.")
    parser.add_argument("--vae-norm-stats-only", action="store_true",
                        help="Skip building the VAE encoder/decoder and only load normalization stats. "
                             "For offline-latent training where latents are pre-extracted; independent of --use-vae "
                             "(which stays True to keep the latent-space model architecture).")
    parser.add_argument("--vae-type", type=str, help="VAE model type.")
    parser.add_argument("--vae-precision", type=str, default="fp32", choices=PRECISIONS,
                        help="Precision mode for the VAE model.")
    parser.add_argument("--vae-autocast-dtype", type=str, default="fp16", choices=PRECISIONS,
                        help="Autocast precision mode for inferring the VAE model.")
    parser.add_argument("--vae-latent-dim", type=int, help="Vae latent dim.")
    parser.add_argument("--vae-spatial-tiling", action="store_true", help="Use spatial tiling for VAE.")
    parser.add_argument("--vae-slicing-bsz", type=int, default=1, help="Vae slicing size.")
    parser.add_argument("--vae-use-compile", action="store_true", help="Use torch.compile to compile vae.")
    parser.add_argument("--vae-prerun-tasks", type=str, nargs="*", help="Tasks to pre-run vae for acceleration.")
    parser.add_argument("--prerun-vae", action="store_true", help="Pre-run vae for acceleration.")
    parser.add_argument("--vae-encode-type", type=str, choices=["sample", "mode"], default="sample",
                        help="Whether to use sampling or mode for vae encoding.")
    parser.add_argument("--noise-dtype", type=str, choices=PRECISIONS, help="Type of noise to use.")

    # vae projector
    parser.add_argument("--patch-size", type=int, help="Patch size for dit-like structures.")
    parser.add_argument("--img-proj-type", type=str, help="Img proj type.")
    parser.add_argument("--patch-embed-hidden-dim", type=int, help="Patch embed hidden dim.")
    parser.add_argument("--video-proj-type", type=str, help="Video proj type.")
    parser.add_argument("--vae-recompute", action="store_true", default=False, help="Whether to recompute the patch embed.")

    # audio vae
    parser.add_argument("--use-audio-vae", action="store_true", help="Whether to use audio VAE model.")
    parser.add_argument("--audio-vae-type", type=str, choices=["waveflow-v1_0", "dual_channel_48k", "dual_channel_48k_refine_decoder"], help="Audio VAE model type.")
    parser.add_argument("--audio-vae-latent-dim", type=int, help="Audio VAE latent dim.")

    # vit
    parser.add_argument("--use-vit", action="store_true", help="Whether to use ViT model.")
    parser.add_argument("--vit-type", type=str, help="ViT type.")
    parser.add_argument("--vit-frozen", action="store_true", help="Whether to freeze the ViT model.")
    parser.add_argument("--vit-precision", type=str, help="Precision mode for the ViT model. Default to the model.")
    parser.add_argument("--vit-recompute", action="store_true", default=False, help="Whether to recompute the ViT model.")
    parser.add_argument("--vit-recompute-num-layers", type=int, default=1, help="Vit recompute num layers")
    
    # vit aligner
    parser.add_argument("--vit-aligner-type", type=str, help="Type of the ViT aligner.")
    parser.add_argument("--vit-aligner-frozen", action="store_true", help="Whether to freeze the ViT aligner.")

    # text encoder
    parser.add_argument("--use-text-encoder", action="store_true", help="Whether to use an extra text encoder.")
    parser.add_argument("--text-encoder-type", type=str, help="Text encoder for dit-like structures.")
    parser.add_argument("--text-encoder-attn", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "flash_attention_3"],
                        help="Attention backend for the full-attention layers of the text encoder "
                             "(only takes effect for transformers-based encoders, e.g. qwen-3.5-9b). "
                             "Set flash_attention_3 to enable FA3.")

    # REPA encoder
    parser.add_argument("--use-repa", action="store_true", help="Whether to use REPA encoder.")
    parser.add_argument("--repa-encoder-type", type=str, help="REPA encoder type.")
    parser.add_argument("--repa-encoder-precision", type=str, help="REPA encoder precision.")
    parser.add_argument("--repa-loss-weight", type=float, default=0.05, help="REPA loss weight.")

    # text loss compute
    parser.add_argument("--text-loss-recompute", action="store_true", default=False, help="Whether to recompute the text loss.")
    parser.add_argument("--compute-text-loss-chunk-num", type=int, default=1,
                        help="Number of the text loss compute chunk. When set gather than 1, combine text-loss-recompute to reduce gpu memory usage.")

    return parser


def add_denoise_core_args(parser: argparse.ArgumentParser, ptm: str | bool = False):
    group = parser.add_argument_group(title="Denoise schedule")

    group.add_argument("--denoise-type", type=str, choices=["ddpm", "flow"], help="Denoise type for noised inputs.")

    # DDPM
    group.add_argument("--ddpm-noise-schedule", type=str, default="scaled_linear",
                       choices=["linear", "scaled_linear", "cosine"],
                       help="Noise schedule for DDPM.")
    group.add_argument("--ddpm-predict-type", type=str, default="v_prediction",
                       choices=["epsilon", "sample", "v_prediction"],
                       help="Prediction type for DDPM.")
    group.add_argument("--enforce-zero-terminal-snr", action="store_true",
                       help="Whether to enforce terminal SNR to 0. Only work when ddpm-predict-type is v_prediction.")
    group.add_argument("--ddpm-learn-sigma", action="store_true", help="Learn sigma for DDPM.")
    group.add_argument("--ddpm-beta-start", type=float, default=0.00085,
                       help="Beta start for DDPM linear or scaled_linear noise schedule.")
    group.add_argument("--ddpm-beta-end", type=float, default=0.012,
                       help="Beta end for DDPM linear or scaled_linear noise scheduler.")
    group.add_argument("--ddpm-noise-offset", type=float, default=0.0,
                       help="Noise offset for DDPM. Add extra channel-wise noise to the input. Useful for learning "
                            "black and white images.")
    group.add_argument("--ddpm-shift-snr", type=float, default=1.0,
                       help="Shift SNR for DDPM. Scale the noise level by this factor. When enlarge training "
                            "resolution (e.g. 512 -> 1024), then shift the SNR by a factor of 2=1024/512. "
                            "Suitable for cosine noise schedule. (Note: not suitable for linear or scaled_linear "
                            "noise schedule.) See the simple-diffusion paper for details.")

    # Flow Matching
    group.add_argument("--flow-path-type", type=str, default="linear", choices=["linear"],
                       help="Path type for flow matching schedulers.")
    group.add_argument("--flow-predict-type", type=str, default="velocity", choices=["velocity"],
                       help="Prediction type for flow matching schedulers.")
    group.add_argument("--flow-loss-weight", type=str, choices=["velocity", "likelihood"],
                       help="Loss weight type for flow matching schedulers.")
    group.add_argument("--flow-train-eps", type=float, help="Small epsilon for avoiding instability during training.")
    group.add_argument("--flow-sample-eps", type=float, help="Small epsilon for avoiding instability during sampling.")
    group.add_argument("--flow-snr-type", type=str, default="lognorm", choices=["uniform", "lognorm", "uniform_lognorm_mix"],
                       help="Type of SNR to use for flow matching schedulers.")
    group.add_argument("--flow-snr-mix-uniform-ratio", type=float, default=0.5,
                       help="Proportion of uniform sampling in uniform+lognorm mix, range [0.0, 1.0]. "
                            "Only used when flow-snr-type is uniform_lognorm_mix. "
                            "0.0 means all lognorm, 1.0 means all uniform.")
    group.add_argument("--flow-shift", type=float, default=1.0,
                       help="Training shift factor for flow matching schedulers.")
    group.add_argument("--flow-reverse", action="store_true", help="If reverse, learning/sampling from t=1 -> t=0.")
    group.add_argument("--flow-solver", type=str, default="euler", choices=["euler"], help="Solver for flow matching.")
    group.add_argument("--flow-start-sigma", type=float, default=1.0, help="Start sigma for flow matching schedulers.")
    group.add_argument("--flow-end-sigma", type=float, default=0.0, help="End sigma for flow matching schedulers.")
    group.add_argument("--sample-flow-shift", type=float,
                       help="[Deprecated] Sampling shift factor for flow matching schedulers.")

    group.add_argument("--flow-shift-video", type=float, default=1.0,
                       help="Shift factor for flow matching schedulers for video data.")
    group.add_argument("--flow-shift-audio", type=float, default=1.0,
                       help="Shift factor for flow matching schedulers for audio data.")

    group.add_argument("--flow-snr-type-video", type=str, default=None,
                       choices=["uniform", "lognorm", "uniform_lognorm_mix"],
                       help="SNR sampling type for video branch. If unset, fall back to --flow-snr-type.")
    group.add_argument("--flow-snr-type-audio", type=str, default=None,
                       choices=["uniform", "lognorm", "uniform_lognorm_mix"],
                       help="SNR sampling type for audio branch. If unset, fall back to --flow-snr-type.")
    group.add_argument("--flow-snr-mix-uniform-ratio-video", type=float, default=None,
                       help="Per-video uniform/lognorm mix ratio. If None, fall back to "
                            "--flow-snr-mix-uniform-ratio. Only used when video snr type is uniform_lognorm_mix.")
    group.add_argument("--flow-snr-mix-uniform-ratio-audio", type=float, default=None,
                       help="Per-audio uniform/lognorm mix ratio. If None, fall back to "
                            "--flow-snr-mix-uniform-ratio. Only used when audio snr type is uniform_lognorm_mix.")
    group.add_argument("--decouple-va-timestep", action="store_true", default=False,
                       help="If set, decouple video/audio timestep sampling in t2va training "
                            "(audio samples its own t instead of reusing video t). Default False keeps "
                            "the original coupled behavior.")

    # Flux shift
    group.add_argument("--use-flux-shift", action="store_true",
                       help="Use flux shift function during training. If enabled, --flow-shift will be ignored.")
    group.add_argument("--flux-base-num-tokens", type=int, default=256, help="Base log shift for flux shift function.")
    # TODO(kevinkhwu): Is this a backward compatibility argument? Should we deprecate it?
    group.add_argument("--flux-base-shift", type=float, default=0.5, help="Base shift for flux shift function.")
    group.add_argument("--flux-base-log-shift", type=float, default=None, help="Base log shift for flux shift function.")
    group.add_argument("--flux-max-num-tokens", type=int, default=4096, help="Max log shift for flux shift function.")
    group.add_argument("--flux-max-shift", type=float, default=1.15, help="Max shift for flux shift function.")
    group.add_argument("--flux-max-log-shift", type=float, default=None, help="Max shift for flux shift function.")
    group.add_argument("--sample-use-flux-shift", action="store_true",
                       help="Use flux shift function during sampling. If enabled, --sample-flow-shift will be ignored.")
    group.add_argument("--use-flux2-shift", action="store_true",
                       help="Use empirical Flux2 shift for mu (overrides --use-flux-shift when set).")
    group.add_argument("--flux2-empirical-num-steps", type=int, default=None,
                       help="num_steps in empirical mu for training; defaults to flow training timesteps (1000).")
    group.add_argument("--sample-use-flux2-shift", action="store_true",
                       help="Use Flux2 empirical shift during sampling (defaults to training --use-flux2-shift if unset).")
    return parser


def add_training_core_args(parser, ptm: str | bool = False):
    if not ptm:
        # Seed and batch size
        parser.add_argument("--micro-batch-size", type=int, default=1,
                            help="Batch size per model instance (local batch size).")
        parser.add_argument("--global-batch-size", type=int,
                            help="global-batch-size = micro-batch-size * world-size * gradient-accumulation-steps")
        # activation recomputation strategy
        parser.add_argument("--recompute-granularity", type=str, choices=["full"],
                            help="Recompute granularity during training. `selective` is not supported yet.")
        parser.add_argument("--recompute-num-layers", type=int, nargs="*", default=[-1],
                            help='The number of Transformer layers to recompute within each pipeline stage. '
                                 '[-1] means all layers in the stage are recomputed. '
                                 'If multiple values are given, they correspond to each pipeline stage respectively.')
        parser.add_argument("--activation-offloading", action="store_true",
                            help="Offload activations to CPU during recompute to reduce GPU memory usage.")
        parser.add_argument("--no-activation-offloading", action="store_false", dest="activation_offloading",
                            help="Disable activation offloading.")
        parser.add_argument("--optimizer-offloading", action="store_true",
                            help="Offload optimizer states to CPU to reduce GPU memory usage.")
        parser.add_argument("--no-optimizer-offloading", action="store_false", dest="optimizer_offloading",
                            help="Disable optimizer offloading.")
        parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping maximum norm.")
        # Checkpointing
        parser.add_argument("--save", type=str, help="Directory to save checkpoints.")
        parser.add_argument("--save-hf", action="store_true", help="Save HuggingFace checkpoint.")
        parser.add_argument("--save-interval", type=int, help="Save interval (in iterations).")
        parser.add_argument("--disable-save-optimizer", action="store_true", help="Disable saving optimizer states.")
        parser.add_argument("--init-save", action="store_true", help="Save the initial model before training.")
        parser.add_argument("--load", type=str, help="Directory to load checkpoint from.")
        parser.add_argument("--load-pretrained-submodules", action="store_true", help="Load pretrained submodules.")
        # Monitor
        parser.add_argument("--log-interval", type=int, default=1, help="Logging interval (in iterations).")
        parser.add_argument("--tensorboard-dir", type=str, help="Directory to write tensorboard logs.")
        parser.add_argument("--wandb-server", type=str, choices=["default", "swanlab"], help="Wandb server to use.")
        parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name.")
        parser.add_argument("--wandb-exp-name", type=str, default="", help="Weights & Biases experiment name.")
        parser.add_argument("--wandb-dir", type=str, default=None, help="Weights & Biases directory.")
        parser.add_argument("--log-timers-to-tensorboard", action="store_true")
        parser.add_argument("--log-memory-to-tensorboard", action="store_true")
        parser.add_argument("--log-num-zeros-in-grad", action="store_true")
        parser.add_argument("--log-params-norm", action="store_true")
        parser.add_argument("--log-throughput", action="store_true")
        parser.add_argument("--use-timer", action="store_true", help="Use timer for detailed speed measurement.")
        # Training schedule
        parser.add_argument("--train-iters", type=int, help="Total number of training iterations.")
        # gc
        parser.add_argument('--manual-gc', action='store_true',
                            help='See mcore arguments.py --manual-gc for more details.')
        parser.add_argument('--manual-gc-interval', type=int, default=0,
                            help='See mcore arguments.py --manual-gc-interval for more details.')
        # Performance
        parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler.")
        # Validation on Training
        parser.add_argument("--val-interval", type=int, default=0, help="Validation interval (in iterations).")
        parser.add_argument(
            "--test-loss-dump-file",
            type=str,
            default=None,
            help="Test-only: if set, dump per-update-step loss list with torch.save.",
        )
        parser.add_argument("--fake-zero-init", action="store_true", help="Fake zero initialization for debugging.")
        parser.add_argument("--init-validation", action="store_true",
                            help="Run validation once before the first training step.")

        parser.add_argument("--try-first-iter", action="store_true", help="Try the first iteration validation before starting the full training.")

    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of steps to accumulate gradients over before performing an update.")
    parser.add_argument("--fsdp-gradient-accumulation-steps", type=int, default=None,
        help=(
            "Number of FSDP communication windows per optimizer update. Must divide "
            "--gradient-accumulation-steps; the default preserves one FSDP "
            "all-gather/reduce-scatter per micro-batch."
        ),
    )
    
    parser.add_argument("--broadcast-data", action="store_true",
                        help="Broadcast batch data from PP=0,CP=0 to all other ranks in the same "
                             "DP group. Required when PP>1 or CP>1 with MoT to guarantee data "
                             "consistency across pipeline stages and context-parallel ranks.")

    parser.add_argument("--dp-load-balance", action="store_true",
                        help="Enable DP load balancing: pre-fetch all micro-batches and "
                             "redistribute across DP ranks via alltoall so that each "
                             "micro-batch step trains the same task type on all ranks.")
    parser.add_argument("--dp-load-balance-task-priority", nargs='*', default=["interleave", "mmu", "t2i", "lm", "pair"])
    parser.add_argument("--dp-load-balance-flops-proxy",
                        choices=["legacy_lsq", "mfu_aligned"],
                        default="legacy_lsq",
                        help="Cost proxy for ranking micro-batches in dp_load_balance. "
                             "'legacy_lsq': sum_i L_i^2 (attention-only). "
                             "'mfu_aligned': compute leo flops's training_total "
                             "(matches mfu log process).")
    parser.add_argument("--dp-load-balance-group-size", type=int, default=0,
                        help="DPLB exchange schedule-group size. 0 (default) uses the full "
                             "DP group and preserves the existing behaviour. A value >1 that "
                             "divides the DP size bounds the reshuffle all_to_all to "
                             "task-stratified subgroups; GBCA still reduces over the full DP "
                             "group so the loss scale is unchanged. ptm_v2 leo backend only "
                             "(the gemini / pure_torch trainers call dp_load_balance_batches "
                             "directly and ignore this).")
    parser.add_argument("--fsdp-encode-balance-group-size", type=int, default=0,
                        help="Maximum DP subgroup size for FSDP image/text buddy encoding "
                             "(--fsdp-encode-balance). 0 (default) uses the full DP group. "
                             "With fixed sampling, deterministic task-MIXED subgroups no larger "
                             "than this value are built, so a modality-heavy rank can borrow "
                             "idle helpers from other tasks while the all-to-all stays bounded.")
    # DP-LB communication optimisation switches (PR-1 ~ PR-6; see
    # DP_LOAD_BALANCE_OPT_README.md). Read at runtime by dp_load_balance.py via
    # get_args(); enable under PERFORMANCE_ARGS.enabled next to dp-load-balance.
    parser.add_argument("--dplb-use-gloo", action="store_true",
                        help="Master gloo switch. Routes dp_load_balance's tiny-msg "
                             "collectives AND the DP-group all_to_all / size all_gather "
                             "through the gloo (CPU/TCP) sibling group; all send/recv "
                             "buffers are created directly on CPU (no GPU H2D/D2H). gloo "
                             "has no native all_to_all, so it is emulated with send/recv. "
                             "Trades GPU/NVLink bandwidth for CPU/TCP; byte-identical "
                             "result. Overrides the NCCL-only two-phase / allgather-meta "
                             "paths.")
    parser.add_argument("--dplb-lean-sizes", action="store_true",
                        help="Replace step-3 O(dp^2) all_gather send-size matrix with an "
                             "O(dp) per-column all_to_all built on the host. Falls back "
                             "when DPLB_USE_ALLGATHER_META is on.")
    parser.add_argument("--dplb-pack-payload", action="store_true",
                        help="PR-5. Pack NUM_DTYPES per-dtype payload a2a into a single "
                             "uint8 a2a (segments size-sorted for zero-copy views). "
                             "Auto-disabled under the two-phase a2a path.")
    parser.add_argument("--dplb-overlap-plan", action="store_true",
                        help="Run the whole next-iter dp-load-balance reshuffle "
                             "(allgather send-plan + all_to_all, CPU/gloo) on a bg "
                             "worker, overlapped with this iter's forward. Requires "
                             "--dplb-use-gloo; incompatible with "
                             "--audio-buddy-encode-balance; skipped on checkpoint iters "
                             "for sampler-resume consistency.")

    # Training batch logging
    parser.add_argument("--saved-training-batch-dir", type=str, help="Directory to save training batches.")
    parser.add_argument("--save-n-training-data", type=int, default=0, help="Save the first N training data.")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--resume-index-batch-sampler", action="store_true",
                        help="Resume IndexBatchSampler state when resuming training. "
                             "It is for online bucketing datasets.")
    parser.add_argument("--no-resume-pack-buffer", action="store_true",
                        help="When resuming with sequence_pack, skip loading pack_buffer. "
                             "Buffer starts empty; minor data skip may occur.")
    parser.add_argument("--save-pack-buffer", action="store_true",
                        help="Save pack_buffer to checkpoint for bitwise-aligned "
                             "resume in sequence_pack mode.")
    parser.add_argument("--deterministic-dataloader", action="store_true",
                        help="Wrap each dataset with DeterministicSeededDatasetWrapper so "
                             "every __getitem__(idx) runs under a stable seed derived from "
                             "(args.seed, dp_rank, dataset_tag, current_epoch, idx). "
                             "Required for bitwise-aligned resume of pack mode (caption "
                             "selection, augmentation, etc.).")
    parser.add_argument("--save-all-ranks-training-states", action="store_true",
                        help="When saving checkpoint, write per-rank "
                             "``training_states/rank{R}.pt`` instead of legacy rank-0-only "
                             "``training_states.pt``")

    # Loss weights
    parser.add_argument("--image-loss-weight", type=float, default=0.0, help="Image loss weight.")
    parser.add_argument("--audio-loss-weight", type=float, default=1.0, help="Audio loss weight.")
    parser.add_argument("--use-global-diffusion-loss-average", action="store_true",
                        help="Average diffusion loss over global image sample count (all ranks). Default: False.")
    parser.add_argument("--use-global-discrete-loss-average", action="store_true",
                        help="Average discrete loss over global token count (all ranks). Default: False.")
    parser.add_argument("--use-global-batch-count-average", action="store_true",
                        help="use global-batch-count to average loss, rather than micro-batch-count, "
                             "only support when dp-load-balance is True now. Default: False.")
    parser.add_argument("--moe-aux-loss", action="store_true", help="Whether to use MoE auxiliary loss.")
    parser.add_argument("--moe-enable-router-expert-bias", action="store_true", help="Whether to enable router expert bias.")
    parser.add_argument("--moe-expert-bias-update-rate", type=float, default=0.001, help="MoE expert bias update rate.")
    parser.add_argument("--moe-score-func", type=str, default="softmax", choices=["sigmoid", "softmax"], help="MoE score function.")
    parser.add_argument("--moe-seq-aux-loss", action="store_true", help="Whether to enable sequence auxiliary loss.")
    parser.add_argument("--moe-enable-expert-bias-zero-mean-update", action="store_true", help="Whether to enable expert bias zero mean update.")
    if not ptm:
        parser.add_argument("--moe-aux-loss-coeff", type=float, default=0.0, help="MoE auxiliary loss weight.")
        parser.add_argument("--vis-moe-expert-status", type=str, default="off", choices=["off", "on"], help="Display MoE expert status")

    parser.add_argument("--per-stage-recompute-num-layers", nargs='+', type=int, default=None,
                        help="Per-stage recompute layer count. Length must equal pipeline-model-parallel-size.")

    return parser


def add_optimizer_core_args(parser, ptm: str | bool = False):
    if not ptm:
        # optimizer
        parser.add_argument("--optimizer", type=str, help="Optimizer type.")
        parser.add_argument("--lr", type=float, help="Learning rate.")
        parser.add_argument("--weight-decay", type=float, help="Weight decay.")
        parser.add_argument("--adam-beta1", type=float, help="Adam beta1.")
        parser.add_argument("--adam-beta2", type=float, help="Adam beta2.")
        parser.add_argument("--adam-eps", type=float, help="Adam eps.")
        parser.add_argument("--momentum", type=float, help="muon momentum.")
        # lr scheduler
        parser.add_argument("--lr-decay-style", type=str, help="Learning rate decay style.")
        parser.add_argument("--lr-warmup-iters", type=int, help="Number of warmup iterations.")
        parser.add_argument("--lr-decay-iters", type=int, help="Number of iterations for learning rate decay.")
        parser.add_argument("--min-lr", type=float, help="Minimum learning rate.")

    return parser


def add_device_mesh_core_args(parser, ptm: str | bool = False):
    parser.add_argument("--init-device", type=str, choices=["cuda", "cpu", "meta"],
                        help="Device to initialize the model.")
    # puretorch related
    if not ptm:
        parser.add_argument("--dp-shard", type=int, default=-1, help="DP shard.")
        parser.add_argument("--dp-replicate", type=int, default=-1, help="DP replicate.")
        parser.add_argument("--local-rank", type=int, help="Local rank for distributed training.")
        parser.add_argument("--tensor-model-parallel-size", type=int, default=1, help="Size of tensor parallel.")
        parser.add_argument("--pipeline-model-parallel-size", type=int, default=1, help="Size of pipeline parallel.")
        parser.add_argument("--pipeline-model-parallel-layout", type=str,
                            help="Layers represented by comma separated numbers, like 1,2,3,4")
        parser.add_argument("--context-parallel-size", type=int, default=1, help="Size of context parallel.")
        parser.add_argument("--expert-model-parallel-size", type=int, default=1, help="Size of expert parallel.")
    return parser


def add_pure_torch_unused_args(parser, ptm: str | bool = False):
    if not ptm:
        parser.add_argument("--tokenizer-type")
        parser.add_argument("--dataloader-type")
        parser.add_argument("--eval-interval", type=int)
        parser.add_argument("--eval-iters", type=int)
        parser.add_argument("--use-flash-attn", action="store_true")
        parser.add_argument("--enable-experimental", action="store_true")
        parser.add_argument("--overlap-grad-reduce", action="store_true")
        parser.add_argument("--overlap-param-gather", action="store_true")
        parser.add_argument("--use-distributed-optimizer", action="store_true")
        parser.add_argument("--ckpt-format")
        parser.add_argument("--dist-ckpt-strictness")
        parser.add_argument("--use-dist-ckpt", action="store_true")
        parser.add_argument("--no-load-optim", action="store_true")
        parser.add_argument("--no-load-rng", action="store_true")
        parser.add_argument("--finetune", action="store_true")
        parser.add_argument("--ckpt-fully-parallel-load", action="store_true")
        parser.add_argument("--ckpt-fully-parallel-save", action="store_true")
        parser.add_argument("--log-validation-ppl-to-tensorboard", action="store_true")
        parser.add_argument("--normalization")
        parser.add_argument("--num-query-groups")
        parser.add_argument("--position-embedding-type")
        parser.add_argument("--rotary-base")
        parser.add_argument("--init-method-std")
        parser.add_argument("--moe-router-topk")
        parser.add_argument("--moe-shared-expert-intermediate-size")
        parser.add_argument("--moe-layer-freq")
        parser.add_argument("--moe-router-load-balancing-type")
        parser.add_argument("--moe-token-dispatcher-type")
        parser.add_argument("--moe-router-dtype")
        parser.add_argument("--attention-dropout", type=float, default=0.0)
        parser.add_argument("--hidden-dropout")
        parser.add_argument("--qk-layernorm", action="store_true")
        parser.add_argument("--group-query-attention", action="store_true")
        parser.add_argument("--swiglu", action="store_true")
        parser.add_argument("--use-mcore-models", action="store_true")
        parser.add_argument("--moe-grouped-gemm", action="store_true")
        parser.add_argument("--use-hunyuan-arch", action="store_true")
        parser.add_argument("--disable-bias-linear", action="store_true")
        parser.add_argument("--moe-enable-deepep", action="store_true")
        parser.add_argument("--moe-permute-fusion", action="store_true")
        parser.add_argument("--moe-router-fusion", action="store_true")
        parser.add_argument("--moe-router-correct-valid-tokens", action="store_true")
        parser.add_argument("--kv-channels")
        parser.add_argument("--dist-ckpt-save-pre-mcore-014", action="store_true")
        parser.add_argument("--distributed-timeout-minutes")
        parser.add_argument("--recompute-method")
        parser.add_argument("--ckpt-step")
        parser.add_argument("--no-save-rng", action="store_true")

        parser.add_argument("--moe-router-bias-update-rate")
        parser.add_argument("--moe-router-score-function")
        parser.add_argument("--embedding-init-method-std", type=float)
        parser.add_argument("--moe-router-topk-scaling-factor", type=float)
        parser.add_argument("--transformer-impl", type=str)
        parser.add_argument("--extra-weight-decay-param-substrings", nargs="*")
        parser.add_argument("--moe-use-legacy-grouped-gemm", action="store_true")
        parser.add_argument("--use-pre-qk-norm", action="store_true")
        parser.add_argument("--untie-embeddings-and-output-weights", action="store_true")
        parser.add_argument("--embedding-no-weight-decay", action="store_true")
        parser.add_argument("--use-unified-init-method", action="store_true")
        parser.add_argument("--moe-router-enable-expert-bias", action="store_true")
        parser.add_argument("--disable-flash-attn-3", action="store_true")
        parser.add_argument("--moe-router-enable-expert-bias-zero-mean-update", action="store_true")
        parser.add_argument("--use-gate-torch-mm", action="store_true")
        parser.add_argument("--muon-ema-gradient", action="store_true")
        parser.add_argument("--muon-momentum", type=float)
        parser.add_argument("--accumulate-allreduce-grads-in-fp32", action="store_true")
        parser.add_argument("--deterministic-mode", action="store_true")
        parser.add_argument("--override-opt_param-scheduler", action="store_true")
        parser.add_argument("--use-dynamic-global-batch-size-convert-iters-to-samples", action="store_true")
        parser.add_argument("--no-persist-layer-norm", action="store_true")
        parser.add_argument("--no-gradient-accumulation-fusion", action="store_true")
        parser.add_argument("--no-masked-softmax-fusion", action="store_true")
        parser.add_argument("--no-bias-swiglu-fusion", action="store_true")
        parser.add_argument("--use-out-place-mask-local-attn", action="store_true")
        parser.add_argument("--use-index-add-local-unpermute", action="store_true")
        parser.add_argument("--deepep-moe-shared-expert-overlap", action="store_true")
        parser.add_argument("--muon-overlap-momentum-gather", action="store_true")
        parser.add_argument("--muon-use-batch-p2p-gather", action="store_true")
        parser.add_argument("--no-align-grad-reduce", action="store_true")
        parser.add_argument("--disable-gloo-process-groups", action="store_true")
        parser.add_argument("--disable-fill-uninitialized-memory", action="store_true")
        parser.add_argument("--ckpt-assume-constant-structure", action="store_true")

        parser.add_argument("--special-lr-mult", type=float, default=1.0)
        parser.add_argument("--special-lr-scale-params", nargs='*', default=None)
        parser.add_argument("--special-lr-scale-exclude-params", nargs='*', default=None)
        parser.add_argument("--special-lr-scale-max-lr", type=float, default=None)
        parser.add_argument("--special-lr-scale-min-lr", type=float, default=None)

        parser.add_argument("--data-path")
        parser.add_argument("--split")
        parser.add_argument("--sft", action="store_true")
        parser.add_argument("--ddp-bucket-size")
        parser.add_argument("--enable-modulate-gate-fusion", action="store_true")
        parser.add_argument("--enable-async-cp", action="store_true")
        parser.add_argument("--enable-fused-norm-merge-rope", action="store_true")
        parser.add_argument("--leo-timing-log-level")
        parser.add_argument("--intra-dp-encode-balance", action="store_true")
        parser.add_argument("--audio-buddy-encode-balance", action="store_true")
        parser.add_argument("--timing-log-level")

        parser.add_argument("--intra-dp-text-pp-stages")
        parser.add_argument("--intra-dp-image-pp-stages")
        parser.add_argument("--intra-dp-encode-by-pp-stage", action="store_true")
        parser.add_argument("--single-dataloader-multi-virtual-pipeline-stages", action="store_true")

        parser.add_argument("--moe-deepep-num-sms")
        parser.add_argument("--activation-offload-pinned-slot", action="store_true")
        parser.add_argument("--pp-multi-tensor-concat", action="store_true")
        parser.add_argument("--get-shapes-from-dataloader", action="store_true")
        parser.add_argument("--recompute-input-offload", action="store_true")
        parser.add_argument("--leo-ep-overlap-activation-offload", action="store_true")
        parser.add_argument("--legacy-gemm-delay-wgrad", action="store_true")

        parser.add_argument("--activation-offload-min-mb")
        parser.add_argument("--moe-valid-tokens-exclude-dummy", action="store_true")
        parser.add_argument("--selective-attn-output-checkpoint-offload", action="store_true")
        parser.add_argument("--pp-multi-tensor-concat-uint8", action="store_true")
        parser.add_argument("--cp-fused-qkv-a2a-max-tokens")
        parser.add_argument("--cp-skip-dummy-branch-a2a", action="store_true")

        parser.add_argument("--recompute-preprocess", action="store_true")

    return parser

def add_ptm_v2_unused_args(parser, ptm: str | bool = False):
    if ptm:
        parser.add_argument("--dp-shard")
        parser.add_argument("--special-lr-mult", type=float, default=1.0)
        parser.add_argument("--special-lr-scale-params", nargs='*', default=None)
        parser.add_argument("--special-lr-scale-exclude-params", nargs='*', default=None)
        parser.add_argument("--special-lr-scale-max-lr", type=float, default=None)
        parser.add_argument("--special-lr-scale-min-lr", type=float, default=None)
    return parser

def add_evaluation_core_args(parser, ptm: str | bool = False):
    parser.add_argument("--ckpt", type=str, help="Model path")
    parser.add_argument("--infer-skip-load-ckpt", action="store_true", help="Skip loading model checkpoint in inference.")
    parser.add_argument("--testsets", type=str, nargs='*', help="Testsets to evaluate.")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity level. 0 for disabled. 1 for sampling info. "
                                                               "2 for streamer logging.")

    parser.add_argument("--bot-task", type=str,
                        choices=["auto", "image", "think", "recaption", "think_recaption", "quickly_think",
                                 "slowly_think", "video", "av", "interleaved", "audio"],
                        help="Type of task for the model. 'auto' for text generation; "
                             "'image' for direct image generation; "
                             "'think' for think->re-write->image; "
                             "'recaption' for re-write->image; "
                             "'think_recaption' for think->re-write->image. "
                             "'quickly_think' for quickly-think->answer; "
                             "'slowly_think' for slowly-think->answer; "
                             "'video' for direct video generation. "
                             "'av' for video-audio generation; "
                             "'interleaved' for interleaved text-image generation. "
                             "'audio' for direct audio generation. "
                             "Default to load from the model generation config.")
    parser.add_argument("--ref-mode", type=str, default=None, choices=["sequence", "channel"],
                        help="Reference image mode for conditional generation. "
                             "'sequence' places cond images as separate sections in the token sequence; "
                             "'channel' concatenates cond images along the latent channel dimension. "
                             "Default to load from the model generation config.")
    parser.add_argument("--generator-device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device for seed generator.")

    # run
    parser.add_argument("--prompt", type=str, nargs="*", help="Prompt to run")
    parser.add_argument("--image-size", type=str, default="auto",
                        help="'auto' means image size is determined by the model. Alternatively, it can be in the "
                             "format of 'HxW' or 'H:W', which will be aligned to the set of preset sizes.")
    parser.add_argument("--use-default-image-size", action="store_true",
                        help="Ignore per-sample height/width from batch and always use --image-size.")
    parser.add_argument("--image", type=str, nargs="*",
                        help="Input image(s) for conditioning. Can be local path, url, or base64 string.")
    parser.add_argument("--num-frames", type=int, default=1, help="Number of frames for video generation.")
    parser.add_argument("--max-new-images", type=int, default=1, help="Number of new images to generate.")
    parser.add_argument("--audio-duration", type=float, help="Duration of generated audio in seconds.")

    # run_testsets
    parser.add_argument("--sample-batch-size", type=int, default=1, help="Batch size for sampling.")
    parser.add_argument("--max-sample-batches", type=int, default=0,
                        help="Maximum number of dataloader batches to run per testset. 0 means no limit.")
    parser.add_argument("--sample-save-base", type=str, help="Base path for saving samples.")

    parser.add_argument("--generation-config", type=str, help="Path to a generation config json file. If None, default"
                                                              "to the config stored in the checkpoint.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of steps to inference for diffusion.")
    parser.add_argument("--diff-guidance-scale", type=float, help="Classifier free guidance scale.")
    parser.add_argument("--diff-guidance-scale-audio", type=float, help="Classifier free guidance scale for audio latent.")
    parser.add_argument("--cfg-distilled", action="store_true", help="Use CFG Distilled when sampling.")
    parser.add_argument("--meanflow", action="store_true", help="Use MeanFlow when sampling.")

    parser.add_argument("--use-system-prompt", type=str, default=None, help="Which system prompt to use during sampling.")
    parser.add_argument("--drop-think-use-system-prompt", type=str, default=None, help="Which system prompt to switch to during drop think sampling.")
    parser.add_argument("--infer-align-image-size", action="store_true", help="Whether to align the target image size to the src image size.")
    parser.add_argument("--skip-existed", action="store_true", help="Whether to skip existed samples.")
    parser.add_argument(
        "--no-interleaved-dummy",
        action="store_true",
        help="Disable dummy interleaved BOI path even when the dataset provides dummy_interleaved_* fields.",
    )
    parser.add_argument(
        "--log-rank0-only",
        action="store_true",
        help="Only emit rank-prefixed logs and streamed text from rank 0; suppress other ranks.",
    )

    # run_eval
    parser.add_argument("--eval-metrics", type=str, nargs='*', help="Evaluation metrics to be evaluated.")
    parser.add_argument("--eval-save-images", action="store_true", help="Whether to save generated images during evaluation.")

    # run_validation_loss
    parser.add_argument(
        "--validation-loss-sets",
        type=str,
        nargs="*",
        help="Validation CSV sets for diffusion validation loss. Format: "
             "path.csv@@plot_name=name [@@caption_col=col ...]",
    )
    parser.add_argument(
        "--validation-loss-timesteps",
        type=int,
        default=8,
        help="Number of intermediate inference timesteps (excluding endpoints) for validation loss.",
    )
    parser.add_argument(
        "--validation-loss-force-rerun",
        action="store_true",
        help="Ignore any cached per-sample validation-loss results for the current iter and recompute "
             "from scratch (deletes SAMPLE_SAVE_DIR/validation_loss/cache|results for this iter).",
    )
    parser.add_argument(
        "--validation-loss-debug-gen",
        action="store_true",
        help="Debug mode: instead of computing the validation loss, run the normal CFG denoising "
             "(self.generate) on the validation-set model_inputs and save the generated video/audio, "
             "to sanity-check that the model_inputs (sequence/masks/rope) are built correctly.",
    )

    # llm
    parser.set_defaults(do_sample=True)
    parser.add_argument("--no-do-sample", action="store_false", dest="do_sample", help="Do not do sampling. Using greedy decoding to process logits.")
    parser.add_argument("--temperature", type=float, help="Temperature for sampling.")
    parser.add_argument("--top-k", type=int, help="Top-K logit.")
    parser.add_argument("--top-p", type=float, help="Top-P logit.")
    parser.add_argument("--repetition-penalty", type=float, help="Repetition penalty.")
    parser.add_argument("--max-new-tokens", type=int, help="Max new tokens to generate.")

    # video
    parser.add_argument("--prompt-prepend-content", type=str, help="Text to prepend to the prompt.")
    parser.add_argument("--prompt-append-content", type=str, help="Text to append to the prompt.")
    parser.add_argument("--prompt-prepend-fps", action="store_true",
                        help="Whether to prepend FPS info to the prompt. The FPS info must be saved in fps column "
                             "of the testset.")
    parser.add_argument("--video-fps", type=int, default=24, help="Video FPS for generating and saving.")

    return parser


def add_rl_core_args(parser, ptm: str | bool = False):
    parser.add_argument("--teacher-checkpoint-dir", type=str, help="Directory to load checkpoint from for teacher model.")
    parser.add_argument("--disable-cp-info", action="store_false", dest="enable_cp_info", help="Disable CP info.")
    return parser

def add_core_args(parser, **kwargs):
    parser.add_argument("--entry", type=str, help="Launcher entry point.")
    parser.add_argument("--task-id", type=str, required=True,
                        help="Task id for determining the identity of current run.")

    parser = add_data_core_args(parser, **kwargs)
    parser = add_model_core_args(parser, **kwargs)
    parser = add_extra_models_core_args(parser, **kwargs)
    parser = add_denoise_core_args(parser, **kwargs)
    parser = add_training_core_args(parser, **kwargs)
    parser = add_optimizer_core_args(parser, **kwargs)
    parser = add_evaluation_core_args(parser, **kwargs)
    parser = add_device_mesh_core_args(parser, **kwargs)
    parser = add_rl_core_args(parser, **kwargs)
    parser = add_pure_torch_unused_args(parser, **kwargs)
    parser = add_ptm_v2_unused_args(parser, **kwargs)
    return parser


# put all args preprocessing here, such as dict key-to-value mapping, default value setting, etc.
def preprocess_args(args):
    # Scope: pure_torch & ptm_v2

    # Parse key-mapping
    if args.key_mapping is not None and len(args.key_mapping) > 0:
        if args.key_mapping[0] in MODEL_KEY_MAPPING:
            args.key_mapping = MODEL_KEY_MAPPING[args.key_mapping[0]]

    # Assign default value for gen-template
    if args.gen_template is None:
        args.gen_template = args.gen_image_template
    else:
        args.gen_image_template = args.gen_template

    if not args.use_mot and not args.gen_template == "multi_stream_dit":
        args.und_token_type = []
        args.gen_token_type = []
        args.audio_token_type = []

    # dplb-overlap-plan runs the FULL reshuffle (allgather + all_to_all) on a
    # background worker during the previous step's forward. That requires the
    # whole path to be CUDA-free (gloo) and free of the GPU-side audio-buddy
    # encode work, so enforce both invariants up front.
    if getattr(args, "dplb_overlap_plan", False):
        assert getattr(args, "dplb_use_gloo", False), (
            "--dplb-overlap-plan requires --dplb-use-gloo: the whole dp-load-balance "
            "reshuffle runs off the main thread and must be CUDA-free."
        )
        assert not getattr(args, "audio_buddy_encode_balance", False), (
            "--dplb-overlap-plan is incompatible with --audio-buddy-encode-balance: "
            "the audio-buddy encode path issues GPU work that cannot run in the "
            "background dp-load-balance worker."
        )
        if int(getattr(args, "dp_load_balance_group_size", 0) or 0):
            # Overlap + stratified groups is supported (the exchange group is created
            # at setup on the main thread, so the worker only hits the cached path).
            # DPLB_VERIFY is not: it rebuilds an all-off NCCL baseline over the FULL
            # DP group to bitwise-compare against, which is not meaningful for a
            # subgroup reshuffle.
            from angelptm.megatron.core.models.leo.dp_load_balance import _DPLB_VERIFY
            assert not _DPLB_VERIFY, (
                "--dplb-overlap-plan + --dp-load-balance-group-size cannot be combined "
                "with DPLB_VERIFY=1: the verify path compares against a full-DP "
                "all-off reshuffle, which a stratified subgroup cannot reproduce."
            )
        assert not getattr(args, "exit_duration_in_mins", None), (
            "--dplb-overlap-plan is incompatible with --exit-duration-in-mins: "
            "the overlap prefetcher primes the NEXT step's reshuffle (a background "
            "gloo all_gather/all_to_all) at the end of each step, and it is only "
            "joined (take()) at the start of the next step. A duration-based exit "
            "breaks the loop at an arbitrary step boundary, leaving that in-flight "
            "collective dangling on the worker thread and risking a hang / mismatched "
            "collective at teardown."
        )
        assert getattr(args, "rerun_mode", "disabled") == "disabled", (
            "--dplb-overlap-plan is incompatible with the rerun engine "
            "(--rerun-mode != 'disabled'): the overlap prefetcher primes the NEXT "
            "step's reshuffle — advancing the data sampler one iteration — at the end "
            "of the current step, BEFORE rerun_state_machine.should_checkpoint_and_exit() "
            "is evaluated. That dynamic checkpoint/rerun request is unpredictable (unlike "
            "the interval-based saves that dplb_will_checkpoint guards), so it cannot be "
            "skipped in time: the checkpoint would capture a sampler state one iteration "
            "ahead, and an in-place rerun would double-advance / desync the single-slot "
            "prefetcher."
        )

    return args


def validate_args(args, defaults=None):
    defaults = default(defaults, {})

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                           with {key}:{v2}'.format(key=key, v=defaults[key], v2=getattr(args, key)),
                      flush=True)
        else:
            setattr(args, key, defaults[key])

    args = preprocess_args(args)

    return args


# ======================================================
#     Define legacy args groups
# ======================================================

def add_logging_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="Logging")

    group.add_argument("--output-dir", type=str, help="Directory to save logs and models")
    if not ptm:
        group.add_argument("--task-flag", type=str, help="Task flag for determining the identity of current run.")
        group.add_argument("--log-every", type=int, default=10, help="Log every N update steps.")
        group.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
        group.add_argument("--profile", action="store_true", help="Enable PyTorch profiler.")
    return parser


def add_tools_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="Tools")

    group.add_argument("--debug", action="store_true", help="Developers should define their own debug behaviors.")
    group.add_argument("--reproduce", action="store_true", help="Enable reproducibility by setting random seeds and deterministic algorithms.")
    return parser


def add_model_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="Model Structure")
    if not ptm:
        group.add_argument("--precision", type=str, choices=PRECISIONS, help="Precision mode for the model.")

    group.add_argument("--model-name", type=str, help="Model name.")
    group.add_argument("--model-structure", type=str, help="Model structure.")
    group.add_argument("--model-kwargs", type=str, help="Model kwargs.")
    group.add_argument("--media-vocab-size", type=int, help="Vocabulary size for media.")
    group.add_argument("--img-proj-type", type=str, help="Img proj type.")
    group.add_argument("--patch-size", type=int, help="Patch size.")
    group.add_argument("--patch-embed-hidden-dim", type=int, help="Patch embed hidden dim.")
    group.add_argument("--rope-type", type=str, default="default", help="Rope type.")
    group.add_argument("--image-preln", action="store_true", help="Image preln.")
    group.add_argument("--unet-out-norm", action="store_true", help="UNet out norm.")
    group.add_argument("--use-recaption-template", action="store_true", help="Use recaption template.")
    group.add_argument("--no-use-recaption-template", action="store_false", dest="use_recaption_template", help="Do not use recaption template.")

    group.add_argument('--use-compile', action='store_true', help='Use torch.compile to compile attn blocks online.')
    group.add_argument('--no-compile', action='store_false', dest='use_compile', help='Do not use torch.compile to compile attn blocks online.')
    group.add_argument('--no-benchmark', action='store_false', dest='benchmark', default=None, help='Disable torch cudnn benchmark mode.')
    group.add_argument('--convert-tp-friendly-qkv', action='store_true', help='Convert qkv to tp friendly format.')
    group.add_argument("--fsdp-impl", type=str, default="streaming", choices=["new", "streaming"],
                       help="FSDP implementation to use in engine.")

    return parser


def add_extra_models_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="Extra Models (VAE, Text Encoder, Tokenizer)")

    # VAE
    group.add_argument("--vae-type", type=str, default=None, help="VAE model type.")
    group.add_argument("--vae-precision", type=str, default="fp32", choices=PRECISIONS, help="Precision mode for the VAE model.")
    group.add_argument("--vae-autocast-dtype", type=str, default=None, choices=PRECISIONS, help="Autocast precision mode for inferring the VAE model.")
    group.add_argument("--vae-latent-dim", type=int, help="Vae latent dim.")
    group.add_argument("--vae-slicing-bsz", type=int, default=1, help="Vae slicing size.")
    group.add_argument("--vae-use-compile", action="store_true", help="Use torch.compile to compile vae.")
    group.add_argument("--vae-prerun-tasks", type=str, nargs="*", help="Tasks to pre-run vae for acceleration.")
    group.add_argument("--prerun-vae", action="store_true", help="Pre-run vae for acceleration.")
    return parser


def add_ema_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="Exponential Moving Average (EMA)")

    if not ptm:
        group.add_argument("--use-ema", action="store_true", help="Enable Exponential Moving Average (EMA).")
        group.add_argument("--ema-decay", type=float,
                        help="Decay factor for EMA. If None, it will be determined by DEFAULT_DECAY. "
                                "See DEFAULT_DECAY for details.")
    group.add_argument("--ema-precision", type=str, default="fp32", choices=PRECISIONS,
                       help="Precision mode for EMA. Options: fp32, bf16. Applied to the EMA model.")
    group.add_argument("--ema-warmup", action="store_true", help="Enable EMA warmup.")
    group.add_argument("--ema-warmup-power", type=float,
                       help="Power factor for EMA warmup. If None, it will be determined by DEFAULT_POWER."
                            "See DEFAULT_POWER for details.")
    group.add_argument("--ema-validation", action="store_true", help="use ema parameters when validation")
    group.add_argument("--distributed-ema", action="store_true", help="Enable Distributed EMA. Only works with `--use-ema`.")
    return parser


# for bc
add_denoise_schedule_args = add_denoise_core_args


# load data args from yaml
def add_data_yaml_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="Data Yaml")
    group.add_argument("--config-path", type=str, help="Path to the yaml config.")
    return parser


# update config in yaml file to args
def parse_data_yaml(args):
    file_path = args.config_path
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataloader_config = config['dataloader_config']
    args.__dict__.update(dataloader_config)
    model_config = config['model_config']
    args.__dict__.update(model_config)

    dataloader_flat_config = flat_dict(dataloader_config, with_parent=True)
    # maybe overwritten, be careful
    args.__dict__.update(dataloader_flat_config)
    model_flat_config = flat_dict(model_config, with_parent=True)
    args.__dict__.update(model_flat_config)


def add_data_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="Data")
    group = add_data_core_args(group, ptm="v1" if ptm else False)

    # type
    group.add_argument("--text-token-length", type=int, help="Text token length.")
    group.add_argument("--image-token-length", type=int, help="Image token length.")

    # Preprocessing
    group.add_argument("--image-size", type=int, nargs='+', help="Image size for training, validation, and sampling. "
                                                                 "If a single value is provided, it will be used for "
                                                                 "both width and height. If two values are provided, "
                                                                 "they will be used for (height, width) respectively.")
    group.add_argument("--training-image-size", type=int, help="Image size for training. If None, default to `image_size`.")
    group.add_argument("--sample-image-size", type=int, nargs='+', help="Image size for sampling. "
                                                                        "If None, default to `image_size`.")
    group.add_argument("--metric-image-size", type=int, nargs='+', help="Image size for metric evaluation. "
                                                                        "If None, default to `image_size`.")
    group.add_argument("--mix-scale", action="store_true", help="Enable mix-scale training. (NotImplemented)")
    group.add_argument("--anchor-size", type=int, nargs='+', default=[256], help="Anchor size for mix-scale training.")
    group.add_argument("--uncond-p", type=float, default=0.1, help="Probability of randomly dropping image description.")

    # Training Dataset
    #   Image
    group.add_argument("--index-file", type=str, nargs='+', help="Index file for the dataset.")
    group.add_argument("--multireso", action="store_true", help="Specify the index-file/val-index-file is a multi-resolution dataset.")
    group.add_argument("--ceph-base", type=str, help="Ceph base")
    group.add_argument("--index-strategy", type=str, default="uniform", choices=INDEX_STRATEGY, help="Sample strategy for multiple index files.")
    group.add_argument("--index-probability", type=float, nargs='+', help="Probability for each index file when index-strategy is probability.")
    group.add_argument("--image-caption-rate", type=float, default=0., help="Set a rate to use caption instead of original description.")
    group.add_argument("--image-text-arrow-suffix", type=str, help="Suffix name for text arrow.")
    group.add_argument("--image-text-col", type=str, help="Column name for text arrow.")
    group.add_argument("--image-caption-arrow-suffix", type=str, help="Suffix name for caption arrow.")
    group.add_argument("--image-caption-col", type=str, default="llava_caption", help="Column name for caption arrow.")
    group.add_argument("--image-token-arrow-suffix", type=str, help="Suffix name for token arrow.")
    group.add_argument("--image-token-col", type=str, help="Column name for token arrow.")
    group.add_argument("--caption-processor", type=str, help="Caption processor for the image captions.")
    group.add_argument("--caption-processor-kwargs", type=str, help="Kwargs for the caption processor.")
    group.add_argument("--caption-sample-ratio", type=str, help="json string format caption-sample-ratio for training, e.g.{\"long_caption\": 1.0, ...}."
                                                                "all keys: long caption, short caption, background, shot type, style, light, atmosphere, camera movement")
    group.add_argument("--image-key", type=str, default="image", help="Column name for image.")
    group.add_argument("--use-general-style", action="store_true", help="Use the `general style` column for constructing training captions.")
    group.add_argument("--try-first-use-source-text", action="store_true",
                       help="Try to use source text for training. Always first check if source_text is available and then apply image_caption_ratio on it.")
    group.add_argument("--raise-text-error", action="store_true", help="Raise exception when catching an error when loading text.")
    group.add_argument("--raise-data-error", action="store_true", help="Raise exception when catching an error when loading data.")
    #   Text
    group.add_argument("--text-index-file", type=str, nargs='+', help="Text index file for the dataset.")
    group.add_argument("--text-sampling-prob", type=float, default=0.1, help="Probability of text-sampling.")
    #   Multimodal Understanding
    group.add_argument("--mmu-index-file", type=str, nargs='+', help="MMU index file for the dataset.")

    # Validation Dataset
    group.add_argument("--val-index-file", type=str, nargs='*', help="Index file for the validation dataset.")
    group.add_argument("--val-multireso", action="store_true", help="Specify the index-file/val-index-file is a multi-resolution dataset.")
    group.add_argument("--val-index-strategy", type=str, default="uniform", choices=INDEX_STRATEGY, help="Sample strategy for multiple index files.")
    group.add_argument("--val-index-probability", type=float, nargs='+', help="Probability for each index file when index-strategy is probability.")
    group.add_argument("--val-image-caption-rate", type=float, default=0., help="Set a rate to use caption instead of original description.")
    group.add_argument("--val-image-text-arrow-suffix", type=str, help="Suffix name for text arrow.")
    group.add_argument("--val-image-text-col", type=str, help="Column name for text arrow.")
    group.add_argument("--val-image-caption-arrow-suffix", type=str, help="Suffix name for caption arrow.")
    group.add_argument("--val-image-caption-col", type=str, default="llava_caption", help="Column name for caption arrow.")
    group.add_argument("--val-caption-sample-ratio", type=str, help="json string format caption-sample-ratio for validation, e.g.{\"long_caption\": 1.0, ...}."
                                                                    "all keys: long caption, short caption, background, shot type, style,light, atmosphere, camera movement")

    parser.add_argument("--dry-run-dataloader", action="store_true", help="Dry run dataloader to check the data loading process.")

    return parser


def add_deepspeed_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="DeepSpeed")
    if ptm:
        return parser

    group.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    group.add_argument("--zero-stage", type=int, default=0, choices=[0, 1, 2, 3],
                       help="DeepSpeed ZeRO stage. 0: off, 1: offload optimizer, 2: offload parameters, "
                            "3: offload optimizer and parameters.")
    group.add_argument("--overlap-comm", action="store_true", help="Enable overlapping communication and computation.")
    group.add_argument("--reduce-bucket-size", type=lambda x: int(float(x)), default=1e9, help="Reduce bucket size for ZeRO.")
    return parser


def add_fsdp_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="FSDP")
    if ptm:
        return parser
    
    group.add_argument("--param-dtype", type=str, default="fp32", choices=PRECISIONS, help="FSDP mixed precision, param dtype")
    group.add_argument("--reduce-dtype", type=str, default="fp32", choices=PRECISIONS, help="FSDP mixed precision, gradient communication precision.")
    group.add_argument("--buffer-dtype", type=str, default="fp32", choices=PRECISIONS, help="FSDP mixed precision, buffer dtype")
    group.add_argument("--sharding-strategy", type=str, default="SHARD_GRAD_OP", choices=SHARDING_STRATEGIES, help="FSDP sharding strategy.")
    group.add_argument("--wrap-module", type=str, default="hymm.models.autoregressive.model.Block", help="FSDP wrap module.")
    return parser


def add_training_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="Training")
    if not ptm:
        group.add_argument("--launcher", type=str, default="deepspeed", help="deepspeed or torch")
        group.add_argument("--micro-batch-size", type=int, default=1,
                           help="Batch size per model instance (local batch size).")
        group.add_argument("--global-batch-size", type=int,
                           help="global-batch-size = micro-batch-size * world-size * gradient-accumulation-steps")
        group.add_argument("--gradient-accumulation-steps", type=int, default=1,
                           help="Number of steps to accumulate gradients over before performing an update.")
        group.add_argument("--global-seed", type=int, default=1, help="Global seed for reproducibility.")
        group.add_argument("--init-save", action="store_true", help="Save the initial model before training.")

    group.add_argument("--skip-nan-grad", action="store_true", help="Skip steps with nan grad when training.")
    group.add_argument("--training-parts", type=str, help="Training a subset of the model parameters.")
    group.add_argument("--final-save", action="store_true", help="Save the initial model before training.")
    group.add_argument("--save-n-training-data", type=int, default=0, help="Save the first N training data.")
    # Load pretrained
    group.add_argument("--pretrained-ckpt", type=str, help="Load pretrained model states.")
    group.add_argument("--load-llm-to-mllm", action="store_true", help="Load llm to mllm.")
    # resume
    group.add_argument("--resume", type=str, help="Checkpoint path to resume training. Can be experiment id or path.")
    group.add_argument("--resume-dataloader", action="store_true", help="Resume dataloader state.")
    group.add_argument("--default-rank0-ss", action="store_true",
                       help="When resuming with GPU number changed, set rank0 to default.")
    group.add_argument("--fix-lr-scheduler", action="store_true",
                       help="Fix the learning rate scheduler when resuming with changed total_num_steps.")
    group.add_argument("--no-load-pretrained", action="store_true",
                       help="Do not load pretrained model states when resuming. Note: refer to args.pretrained_ckpt")
    group.add_argument("--no-load-optim-states", action="store_true",
                       help="Do not load optimizer states when resuming.")
    group.add_argument("--no-load-ema", action="store_true", help="Do not load EMA states when resuming.")
    # Mix scale training
    group.add_argument("--mix-micro-batch-size", type=int, default=[1], nargs='+',
                       help="Mix micro batch size for training.")

    group.add_argument("--max-epochs", type=int, default=100, help="Number of epochs to train.")
    group.add_argument("--max-training-steps", type=int, default=10_000_000, help="Number of steps to train.")
    group.add_argument("--ckpt-every", type=int, default=5000, help="Save checkpoint every a few update steps.")

    # Acceleration
    group.add_argument("--gradient-checkpoint", action="store_true",
                       help="Enable gradient checkpointing to reduce memory usage.")
    group.add_argument("--gradient-checkpoint-layers", type=int, default=-1,
                       help="Number of layers to checkpoint. -1 for all layers. `n` for the first n layers.")

    # Loss weights
    group.add_argument("--image-loss-weight", type=float, help="Image loss weight.")

    # Validation on training
    group.add_argument("--validation-every", type=int, default=5000, help="Evaluate every N update steps.")
    group.add_argument("--validation-at-steps", type=int, default=[100], nargs='*',
                       help="Evaluate at specific update steps during training. It is useful to evaluate the model at early steps to check if everything is working properly.")
    group.add_argument("--validation-metrics", type=str, default=None, nargs='*',
                       help="Validation types to evaluate during training.")
    group.add_argument("--validation-metrics-save-image", action="store_true", help="Save evaluation images.")

    # puretorch related
    group.add_argument("--ep-size", type=int, default=1, help="Size of expert parallel.")
    group.add_argument("--pp-size", type=int, default=1, help="Size of pipeline parallel.")
    parser.add_argument("--pp-splits", type=str, default=None, help="1,2,3,4这种")

    return parser


def add_evaluation_args(parser: argparse.ArgumentParser, ptm: bool = False):
    group = parser.add_argument_group(title="Evaluation")

    # ======================== Common arguments ========================
    group.add_argument("--evaluation-media-type", type=str, default="image", choices=["image", "audio"], help="Media type of evaluation, default is image")
    group.add_argument("--eval-batch-size", type=int, help="Batch size for inference.")
    group.add_argument("--sample-batch-size", type=int, help="Batch size for sampling. Default to `--eval-batch-size`.")
    group.add_argument("--metric-batch-size", type=int, help="Batch size for metric evaluation. Default to `--eval-batch-size`.")
    group.add_argument("--infer-steps", type=int, default=16, help="Number of steps to inference.")
    group.add_argument("--diff-infer-steps", type=int, help="Number of steps to inference for diffusion head. If None, default to `infer_steps`.")
    group.add_argument("--infer-align-image-size-mode", type=str, choices=["resize", "resize_and_pad"], help="How to align the target image size to the src image size.")
    group.add_argument("--pipeline", type=str, help="Pipeline for inference.")
    group.add_argument("--task", type=str, default="sample", help="Inference task.")
    group.add_argument("--tp-size", type=int, default=1, help="TP size.")
    # use hf inference
    parser.add_argument("--use-hf", action="store_true", help="use huggingface model")

    # Eval launcher
    if not ptm:
        group.add_argument("--ddp", action="store_true", help="Enable Distributed Data Parallel (DDP) during sampling.")
        group.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for DDP.")
        group.add_argument("--node-index", type=int, default=0, help="Node index for DDP.")
        group.add_argument("--deepspeed", action="store_true", help="Enable Deepspeed during sampling.")
    group.add_argument("--use-ptm", action="store_true", help="use ptm in model")
    # ckpt
    group.add_argument("--ckpt", type=str, help="Path to the checkpoint to evaluate.")
    group.add_argument("--load-key", type=str, default="module", choices=["module", "ema"], help="Key to load the model states.")
    group.add_argument("--weights-only", action="store_true", help="Passed to torch.load to load only the model weights.")
    group.add_argument("--no-load-model", action="store_true", help="Do not load the model states.")
    group.add_argument("--args-path", type=str, help="Path to the args.json file.")
    group.add_argument("--no-load-pretrained-vision-model", action="store_true", help="Do not load pretrained vision model to accelerate model building.")

    group.add_argument("--puretorch-ckpt", type=str, help="")

    # Sampling
    group.add_argument("--predict-image-shape-token", action="store_true", help="Predict the image shape token during sampling.")
    if not ptm:
        group.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
        group.add_argument("--top-k", type=int, default=1024, help="[ManualAssign] Top-K logit.")
        group.add_argument("--top-p", type=float, default=0.9, help="[ManualAssign] Top-P logit.")
        group.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum number of new tokens to generate.")
    # MLM
    group.add_argument("--schedule-type", type=str, default="arccos",
                       choices=["linear", "root", "square", "cosine", "arccos", "shift"], help="Schedule type for MLM.")
    group.add_argument("--mask-ratio-shift", type=float, default=1.0, help="Shift factor for MLM schedule.")

    # Data
    group.add_argument("--eval-data-repeat-times", type=int, default=0, help="Repeat times for evaluation data. It is useful for text generation when enabling expert parallel (EP>1).")

    # ======================== Test-time improvements ========================
    group.add_argument("--kv-cache", action="store_true", help="Use KV Cache when sampling.")
    group.add_argument("--block-size", type=int, help="Maximum sequence size.")
    group.add_argument("--rope-base-rescale-factor", type=float, nargs='+', default=1.0, help="NTK rescale factor for rope base.")
    group.add_argument("--infer-attn-type", type=str, choices=["auto", "flex"], default="auto", help="Inference attention type.")
    # Classifier-free guidance
    group.add_argument("--cfg-mode", type=str, default=None, choices=["cfg", "apg", "mixed_cfg_apg"], help="Classifier-free guidance mode.")
    group.add_argument("--adaptive-projected-guidance-rescale", type=float, default=10.0, help="In mixed-cfg-apg mode, rescale the guidance scale for apg.")
    group.add_argument("--apply-cfg-timesteps-ratio", type=float, default=0.1, help="In mixed-cfg-apg mode, apply cfg at the first N timesteps.")
    group.add_argument("--guidance-scale", type=float, default=6.0, help="[ManualAssign] Classifier free guidance scale.")
    group.add_argument("--face-guidance-scale", type=float, default=1.0, help="[ManualAssign] Classifier free guidance scale for face.")
    group.add_argument("--src-guidance-scale", type=float, default=1.0, help="[ManualAssign] Classifier free guidance scale for src image.")
    group.add_argument("--neg-prompt", type=str, help="Negative prompt for sampling.")
    group.add_argument("--guidance-dynamic", type=str, choices=["const", "linear_inc", "linear_dec"],
                       help="Dynamic guidance scale.")
    # Language mode
    if not ptm:
        group.add_argument("--do-sample", action="store_true", help="Do sampling. Using logits processors to process logits.")
    group.add_argument("--no-do-sample", action="store_false", dest="do_sample", help="Do not do sampling. Using greedy decoding to process logits.")
    group.add_argument("--predict-image-shape-no-do-sample", action="store_true", help="Do not do sampling when predicting image shape token.")
    group.add_argument("--cot-max-length", type=int, default=0, help="Maximum length for chain-of-thought (CoT) generation. If 0, no CoT generation.")
    group.add_argument("--t2i-pred-text-mode", type=str, choices=["off", "cot_think", "cot_recaption", "mixed"], help="Predict text mode before image generation.")
    group.add_argument("--eval-stop-by-dummy", action="store_true", help="Stop evaluation by dummy token across all ranks.")
    # Face
    group.add_argument("--load-insightface", action="store_true", help="Load insightface model.")
    group.add_argument("--verbose", type=int, default=1, help="Verbosity level.")
    # MMU
    group.add_argument("--mmu-prompt", type=str, help="MMU prompt for inference.")
    group.add_argument("--legacy-t2i", type=str, help="Legacy T2i model for inference.")
    # LM
    group.add_argument("--drop-think", action="store_true", help="Drop the CoT thinking part before entering t2i.")
    group.add_argument("--only-recaption", action="store_true", help="Break after recaption in CoT T2I.")
    group.add_argument("--gate-precision", type=str, default="fp32", choices=["fp32", "bf16"], help="Precision for the gating layer.")
    # Backward compatibility
    group.add_argument("--fix-bc", type=str, nargs='+', default=[], help="Fix backward compatibility issues for given keys.")
    # MoE
    group.add_argument("--no-drop-tokens", action="store_true", help="Use unlimited capacity for MoE during inference.")
    group.add_argument("--infer-use-fused-moe", action="store_true", help="Use fused MoE for inference.")
    group.add_argument("--infer-use-flash-attn", action="store_true", help="Use flash attn for inference.")
    group.add_argument("--capacity-factor", type=float, default=1.0, help="Capacity factor for MoE during inference.")

    # Seed
    if not ptm:
        group.add_argument("--seed-type", type=str, default="auto", choices=["file", "random", "fixed", "auto"], help="""
                            Seed type for evaluation.
                                - file: Use the seed from the CSV file.
                                - random: Generate a random seed.
                                - fixed: Use the fixed seed given by `--seed`.
                                - auto:
                                  * In `csv` mode, it will use the seed column if available, otherwise use the fixed `seed` value.
                                  * In `prompt` mode, it will use the fixed `seed` value.
                            """)
        group.add_argument("--seed", type=int, default=123456, help="Seed for evaluation.")

    # Size
    group.add_argument("--exact-size", action="store_true", help="Use exact size for sampling.")

    # ======================== Validation metrics ========================
    group.add_argument("--save-path", type=str, help="Path to save the evaluation results.")
    group.add_argument("--save-path-suffix", type=str, default="", help="Suffix for the path of saved file.")
    group.add_argument("--save-suffix", type=str, default="", help="Suffix for the names of saved file.")

    group.add_argument("--metric-save-path", type=str, help="Path to save the evaluation metric results. Default to `--save-path`.")
    group.add_argument("--metric-save-file-suffix", type=str, help="Suffix for the names of saved file. Default to `--save-suffix`.")

    group.add_argument("--evaluation-metrics", type=str, nargs='*', help="Evaluation metrics to be evaluated.")
    group.add_argument("--evaluation-metrics-save-image", action="store_true", help="Save images for evaluation metrics.")

    group.add_argument("--rerank", type=int, default=1, help="Rerank times for evaluation metrics.")

    # ======================== Text/Image/Video/Audio Generation ========================
    group.add_argument("--save-dir", type=str, help="Path to save the sampling results.")
    group.add_argument("--save-dir-suffix", type=str, default="", help="Suffix for the dir of sampling results.")
    group.add_argument("--sample-save-base", type=str, help="Base path to save the generated samples, a subdirectory will be created.")
    group.add_argument("--sample-save-path", type=str, help="Path to save the generated samples. Default to `--save-dir`.")
    group.add_argument("--sample-save-path-suffix", type=str, help="Suffix for the path of saved samples. Default to `--save-dir-suffix`.")
    group.add_argument("--sample-save-file-suffix", type=str, help="Suffix for the names of saved samples. Default to `--save-suffix`.")

    group.add_argument("--num-sample-per-prompt", type=int, default=1, help="Number of samples to generate for each prompt.")
    # --- interactive ---
    group.add_argument("--interactive", action="store_true", help="Interactive mode, the program will ask for one prompt each time.")
    group.add_argument("--prompt", type=str, help="Prompt for sampling during evaluation.")
    group.add_argument("--json-input", type=str, help="A JSON file of a message list containing context and question for inference.")
    group.add_argument("--label", type=int, help="ImageNet label for sampling.")
    group.add_argument("--n-class", type=int, help="Number of classes for sampling.")
    group.add_argument("--image", type=str, help="Image path for sampling.")
    group.add_argument("--mask", type=str, help="Mask path for sampling.")
    group.add_argument("--question", type=str, help="User question for LM inference.")
    group.add_argument("--stream", action="store_true", help="Stream output.")
    # --- csv ---
    group.add_argument("--csv", type=str, help="CSV file for evaluation.")
    group.add_argument("--skip-exist", action="store_true", help="Skip existed samples.")
    group.add_argument("--testsets", type=str, nargs='*', help="Testsets to evaluate. If specified, it is prioritized over `--csv` and `--task`.")
    # --- instruction tuning ---
    group.add_argument("--instruction-tuning-task", type=str, help="The instruction tuning task which you want to evaluate.")
    group.add_argument("--subject-driven-shape-index", type=int, default=-1, help="Index that indicates which src image's shape that target image should follow. Default to -1, which means the target image will follow the last src image's shape.")

    # ======================== Taylor Cache ========================
    group.add_argument("--use-taylor-cache", action="store_true", help="Use Taylor Cache when sampling.")
    group.add_argument("--cfg-distilled", action="store_true", help="Use CFG Distilled when sampling.")
    group.add_argument("--taylor-cache-interval", type=int, default=5, help="Interval of Taylor Cache.")
    group.add_argument("--taylor-cache-order", type=int, default=2, help="Order of Taylor Cache.")
    group.add_argument("--taylor-cache-enable-first-enhance", action="store_true", help="Enable first enhance when using Taylor Cache.")
    group.add_argument("--taylor-cache-first-enhance-steps", type=int, default=3, help="First enhance steps when using Taylor Cache (>2).")
    group.add_argument("--taylor-cache-enable-tailing-enhance", action="store_true", help="Enable tailing enhance when using Taylor Cache.")
    group.add_argument("--taylor-cache-tailing-enhance-steps", type=int, default=1, help="Tailing enhance steps when using Taylor Cache.")
    group.add_argument("--taylor-cache-low-freqs-order", type=int, default=2, help="Low freqs order when using Taylor Cache.")
    group.add_argument("--taylor-cache-high-freqs-order", type=int, default=2, help="High freqs order when using Taylor Cache.")    
    return parser


def recursive_subkey_to_object(obj, root=True):
    subkey_to_object = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            subkey_to_object.update(recursive_subkey_to_object(value, False))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, dict):
                    subkey_to_object.update(recursive_subkey_to_object(v, False))
        elif not root:
            subkey_to_object[key] = obj
    return subkey_to_object


def manual_assign(args, remain_args):
    # Get subkey to object mapping
    subkey_to_object = recursive_subkey_to_object(vars(args))
    for arg in remain_args:
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if key in subkey_to_object:
                subkey_to_object[key][key] = getattr(args, key)
    return args


def deprecated_warning(args, deprecated_arg, new_arg=None):
    deprecated_arg = deprecated_arg.lstrip("-").replace("-", "_")
    if hasattr(args, deprecated_arg) and getattr(args, deprecated_arg) is not None:
        if new_arg:
            logger.warning(f"Argument `{deprecated_arg}` will be deprecated, please use `{new_arg}` instead.")
        else:
            logger.warning(f"Argument `{deprecated_arg}` will be deprecated.")


def sanity_check_args(args):
    '''
    参数合法性总入口：把所有“用户随手填的”超参全部检查、补全、对齐一遍，
    一旦有不一致或过期字段，当场抛异常或打 warning ，避免训练到一半才爆雷。

    职能清单：
    1. 图像尺寸支持“单值/多值/成对”写法，自动拆成 (宽,高) 列表。
    2. 把缺省字段一次性补齐， 防止下游代码到处“if args.xxx is None”。
    3. 多尺度训练时，保证 anchor 数量与 micro-batch 数量严格对齐。
    4. VAE 类型与词表大小强制对齐，避免 codebook 与 embedding 对不上。
    5. 互斥开关检查（ iw/ih token vs shape token ）。
    6. 过期字段扫描，提示用户迁移到新 flag。

    返回：
    args —— 所有字段已补齐、对齐、断言完毕的 args 对象。
    '''

    # Image size
    training_image_size = as_tuple(args.training_image_size)
    if len(training_image_size) > 2:
        assert len(training_image_size) % 2 == 0, (
            f"Invalid image size: {training_image_size}. If more than two values are provided, each pair of values will be "
            f"used for width and height respectively. The sampling process will be run for each pair of values."
        )

    # Set default values
    args.training_image_size = default(args.training_image_size, args.image_size)
    args.sample_image_size = default(args.sample_image_size, args.image_size)
    args.metric_image_size = default(args.metric_image_size, args.image_size)
    args.sample_batch_size = default(args.sample_batch_size, args.eval_batch_size)
    args.metric_batch_size = default(args.metric_batch_size, args.eval_batch_size)
    args.diff_infer_steps = default(args.diff_infer_steps, args.infer_steps)
    args.metric_save_path = args.metric_save_path or args.save_path
    args.metric_save_file_suffix = args.metric_save_file_suffix or args.save_suffix
    args.sample_save_path = args.sample_save_path or args.save_dir
    args.sample_save_path_suffix = args.sample_save_path_suffix or args.save_dir_suffix
    args.sample_save_file_suffix = args.sample_save_file_suffix or args.save_suffix
    args.model_structure = default(args.model_structure, getattr(args, 'model_type', None))
    if hasattr(args, 'model_precision') and args.precision != args.model_precision:
        logger.warning(f"\n*************************************************************************"
                       f"\n    `precision` is {args.precision} but `model_precision` is {args.model_precision}."
                       f"\n*************************************************************************")

    if args.autocast_dtype is None:
        logger.warning(f"You set autocast_dtype as None, we will use autocast_dtype same with precision={args.precision}. Please make sure this is what you want.")
        args.autocast_dtype = args.precision
    if args.vae_autocast_dtype is None:
        logger.warning(f"You set vae_autocast_dtype as None, we will use vae_autocast_dtype same with vae_precision={args.vae_precision}. Please make sure this is what you want.")
        args.vae_autocast_dtype = args.vae_precision

    # Mix-scale training/validation
    mix_micro_batch_sizes = as_tuple(args.mix_micro_batch_size)
    anchor_sizes = as_tuple(args.anchor_size)
    if len(mix_micro_batch_sizes) != len(anchor_sizes):
        raise ValueError(f"Number of micro batch sizes ({len(mix_micro_batch_sizes)}) must match the number of anchor "
                         f"sizes ({anchor_sizes}).")

    # vae
    if args.vae_type is not None:
        vae_meta_info = VAE_META_INFO[args.vae_type]
        if "codebook_size" in vae_meta_info:
            assert args.media_vocab_size == vae_meta_info["codebook_size"], (
                f"media_vocab_size ({args.media_vocab_size}) must match the vae codebook size ({vae_meta_info['codebook_size']})."
            )

    assert not (getattr(args, "use_flux_shift", False) and getattr(args, "use_flux2_shift", False)), (
        "use_flux_shift and use_flux2_shift cannot both be True"
    )

    assert not (args.add_iw_ih_token and (args.add_image_shape_token or args.add_tw_th_token)), \
        "add_iw_ih_token and add_image_shape_token/add_tw_th_token cannot be both True"
    assert not (args.add_image_shape_token and args.add_tw_th_token), \
        "add_image_shape_token and add_tw_th_token cannot be both True"

    if args.multireso ^ (args.add_iw_ih_token or args.add_image_shape_token or args.add_tw_th_token):
        logger.warning(f"You set multireso as {args.multireso} but add_iw_ih_token as {args.add_iw_ih_token} or add_image_shape_token as {args.add_image_shape_token} or add_tw_th_token as {args.add_tw_th_token}. Please make sure this is what you want.")

    # Deprecation
    deprecated_warning(args, "lr_scheduler_name", "--lr-schedule")
    deprecated_warning(args, "model_type", "--model-structure")
    deprecated_warning(args, "model_precision", "--precision")

    return args
