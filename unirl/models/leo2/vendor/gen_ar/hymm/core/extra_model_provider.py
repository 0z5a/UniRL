# Define extra components here
#
import torch
from argparse import Namespace

from .global_vars import (
    get_args,
    get_parallel_state,
    get_logger,
    get_vae,
    get_audio_vae,
    get_combined_iterator,
    get_mm_state,
    set_vae,
    set_audio_vae,
    set_denoiser,
    set_video_denoiser,
    set_audio_denoiser,
    set_tkwrapper,
    set_scalar_state,
    set_text_encoder,
    set_repa_encoder,
)


def build_tkwrapper():
    from hymm.models.tokenizers import load_tokenizer

    args = get_args()
    tkwrapper = load_tokenizer(args.tokenizer_name, args.tokenizer_class)
    set_tkwrapper(tkwrapper)
    return tkwrapper


def calc_vae_prerun_sizes():
    from index_kits import MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2

    ds_state = get_mm_state().distributed_sampling_state
    _iterator = get_combined_iterator()
    datasets, dataloaders = _iterator.datasets, _iterator.dataloaders

    # Pre-run vae arguments. Each is a tuple of (batch_size, height, width)
    vae_prerun_sizes = set()
    cur_tasks = list(datasets.keys()) if ds_state.cur_key is None else [ds_state.cur_key]
    for task in cur_tasks:
        dataset = datasets[task]
        dataloader = dataloaders[task]
        if dataloader.batch_size is None:
            bsz = dataloader.batch_sampler.batch_size
        else:
            bsz = dataloader.batch_size
        if isinstance(dataset.index_manager, MultiResolutionBucketIndexV2):
            for bucket in dataset.index_manager.buckets:
                vae_prerun_sizes.add((bsz, bucket.height, bucket.width))
        elif isinstance(dataset.index_manager, MultiMultiResolutionBucketIndexV2):
            for multireso in dataset.index_manager.buckets:
                for bucket in multireso.buckets:
                    vae_prerun_sizes.add((bsz, bucket.height, bucket.width))
        elif hasattr(dataset, "vae_reso_group"):
            for reso in dataset.vae_reso_group.data:
                vae_prerun_sizes.add((bsz, reso.h, reso.w))
                # Always add batch size 1 for list of images scenario
                vae_prerun_sizes.add((1, reso.h, reso.w))
            vae_prerun_sizes.add((bsz, dataset.vae_info.h_factor, dataset.vae_info.w_factor))
            vae_prerun_sizes.add((1, dataset.vae_info.h_factor, dataset.vae_info.w_factor))
        else:
            vae_prerun_sizes.add((bsz, dataset.vae_info.h_factor, dataset.vae_info.w_factor))
            vae_prerun_sizes.add((1, dataset.vae_info.h_factor, dataset.vae_info.w_factor))

    return vae_prerun_sizes


@torch.no_grad()
def prerun_vae(model=None):
    from ..utils.torch_utils import PRECISION_TO_TYPE

    args = get_args()

    if not getattr(args, "use_vae", True) or getattr(args, "vae_norm_stats_only", False):
        if get_parallel_state().dp_rank == 0:
            from loguru import logger
            logger.info("Skip pre-running VAE (model has no VAE or norm-stats-only mode).")
        torch.distributed.barrier()
        return

    mm_state = get_mm_state()
    device = torch.device("cuda", args.local_rank)
    prerun_model_conv = model is not None and model._config.img_proj_type == "conv" # noqa

    if mm_state.distributed_sampling_state.cur_rank == 0:
        from loguru import logger
    else:
        from ..utils.file_utils import empty_logger
        logger = empty_logger()

    vae = get_vae()
    vae_prerun_sizes = sorted(list(calc_vae_prerun_sizes()), key=lambda x: (x[1], -x[2]))
    conv_prerun_sizes = set()
    from .global_vars import set_prerun_sizes
    set_prerun_sizes(vae_prerun_sizes, conv_prerun_sizes)

    torch.distributed.barrier()
    if len(vae_prerun_sizes) > 0:
        logger.info("Pre-running VAE...")

        vae_autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]
        with torch.autocast(
                device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32
        ):
            for i, (bsz, height, width) in enumerate(vae_prerun_sizes):
                logger.info(f"    Bucket {i:>3d}: {bsz}x3x{height}x{width}")
                image = torch.rand((bsz, 3, height, width), device=device)
                # pre-running vae
                output = vae.encode(image)
                # pre-running conv layers in model
                if prerun_model_conv:
                    latents = output.latent_dist.mode()
                    if hasattr(vae, "ffactor_temporal"):
                        assert latents.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
                        latents = latents.squeeze(2)
                    t_emb = torch.ones((bsz, model._config.hidden_size), device=device) # noqa
                    conv_prerun_sizes.add(tuple(latents.shape))
                    image_seq, token_height, token_width = model.patch_embed(latents, t_emb)
                    conv_prerun_sizes.add(tuple(image_seq.shape) + (token_height, token_width))
                    _ = model.final_layer(image_seq, t_emb, token_height, token_width)
    else:
        logger.info("No task relies on VAE. Skip pre-running VAE.")

    torch.distributed.barrier()


def build_vae(dp_rank=None, only_encoder=True, norm_stats_only=False):
    from ..models.autoencoders import load_vae

    args = get_args()
    logger = get_logger()
    device = torch.device("cuda", args.local_rank)

    if not norm_stats_only:
        logger.info("Building VAE...")
        vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=device,
            logger=logger,
            args=args,
            only_encoder=only_encoder,
        )
        if args.vae_spatial_tiling:
            vae.enable_spatial_tiling()
        if args.vae_slicing_bsz is not None and args.vae_slicing_bsz > 1:
            vae.slicing_bsz = args.vae_slicing_bsz
            vae.enable_slicing()
            logger.info(f"VAE slicing batch size set to {vae.slicing_bsz}")
        if args.vae_use_compile:
            vae.use_compile = args.vae_use_compile
            logger.info(f"VAE use_compile set to {vae.use_compile}")
    else:
        logger.info("Skip building VAE encoder/decoder, loading norm stats only (offline latent mode)...")
        from ..models.autoencoders import configure_vae_latent_normalization
        from ..constants import VAE_META_INFO
        from ..utils.torch_utils import PRECISION_TO_TYPE

        vae_meta_info = VAE_META_INFO[args.vae_type]
        vae_path = vae_meta_info["path"]
        vae = Namespace()
        vae.autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]
        configure_vae_latent_normalization(vae, args=args, logger=logger, vae_path=vae_path)

    if dp_rank is None:
        dp_rank = get_parallel_state().dp_rank
    if getattr(args, "intra_dp_encode_balance", False):
        vae.generator = torch.Generator(device).manual_seed(args.seed + torch.distributed.get_rank()) # global rank
    else:
        vae.generator = torch.Generator(device).manual_seed(args.seed + dp_rank)
    vae.noise_generator = torch.Generator(device).manual_seed(args.seed + dp_rank)

    set_vae(vae)
    return vae


def build_audio_vae(dp_rank=None, only_encoder=False, use_audio_vae=True):
    from ..models.audio_encoders import load_audio_encoder

    args = get_args()
    logger = get_logger()
    device = torch.device("cuda", args.local_rank)

    if use_audio_vae:
        logger.info("Building audio VAE...")
        audio_vae = load_audio_encoder(
            args.audio_vae_type,
            audio_vae_latent_dim=args.audio_vae_latent_dim,
            device=device,
            only_encoder=only_encoder,
        )
    else:
        logger.info(f"Skip load audio VAE.")
        audio_vae = Namespace()

    if dp_rank is None:
        dp_rank = get_parallel_state().dp_rank

    # TODO: Maybe use the same starting generator in inference has some benefits in terms of
    #  training-inference consistency?
    if getattr(args, "intra_dp_encode_balance", False):
        audio_vae.generator = torch.Generator(device).manual_seed(args.seed + torch.distributed.get_rank()) # global rank
    else:
        audio_vae.generator = torch.Generator(device).manual_seed(args.seed + dp_rank)
    audio_vae.noise_generator = torch.Generator(device).manual_seed(args.seed + dp_rank)

    set_audio_vae(audio_vae)
    return audio_vae


def build_denoiser(denoiser_type="image", **kwargs):
    from hymm.diffusion import load_denoiser

    args = get_args()
    denoiser = load_denoiser(args, **kwargs)

    if denoiser_type == "image":
        set_denoiser(denoiser)
    elif denoiser_type == "video":
        set_video_denoiser(denoiser)
    elif denoiser_type == "audio":
        set_audio_denoiser(denoiser)
    else:
        raise ValueError(f"Unknown denoiser type: {denoiser_type}")

    return denoiser


def build_scalar_state(scalar_state=None):
    from hymm.trainers.helpers import MultiModalScalarStates

    if scalar_state is None:
        scalar_state = MultiModalScalarStates()
    else:
        scalar_state = MultiModalScalarStates.from_pretrained(scalar_state)
    set_scalar_state(scalar_state)
    return scalar_state


def build_text_encoder(fsdp_mesh=None):
    from hymm.models.text_encoder import TextEncoder
    from hymm.constants import LI_DIT_PROMPT_TEMPLATE

    args = get_args()
    logger = get_logger()
    device = torch.device("cuda", args.local_rank)

    if hasattr(args, "text_encoder_prompt_template") and \
        args.text_encoder_prompt_template is not None and \
        args.text_encoder_prompt_template in LI_DIT_PROMPT_TEMPLATE:
        import warnings
        warnings.warn(
            "'text_encoder_prompt_template' is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        prompt_template = LI_DIT_PROMPT_TEMPLATE[args.text_encoder_prompt_template]
    else:
        prompt_template = True  # deprecated, any non-None value suffices
        
    text_encoder = TextEncoder(
        text_encoder_type=args.text_encoder_type,
        max_length=args.text_encoder_text_len,
        text_encoder_precision=args.text_encoder_precision,
        prompt_template=prompt_template,
        hidden_state_skip_layer=args.text_encoder_hidden_state_skip_layer,
        logger=logger,
        device=device,
        uncond_token=args.text_encoder_uncond_token,
        use_fsdp=getattr(args, "text_encoder_use_fsdp", False),
        fsdp_overlap=getattr(args, "text_encoder_fsdp_overlap", False),
        fsdp_mesh=fsdp_mesh,
        attn_implementation=getattr(args, "text_encoder_attn", None),
    )
    set_text_encoder(text_encoder)
    return text_encoder


def build_repa_encoder():
    from hymm.models.visual_encoders.repa import RepaEncoder

    args = get_args()
    logger = get_logger()
    device = torch.device("cuda", args.local_rank)

    repa_encoder = RepaEncoder(
        repa_encoder_type=args.repa_encoder_type,
        repa_encoder_precision=args.repa_encoder_precision,
        logger=logger,
        device=device
    )
    set_repa_encoder(repa_encoder)
    return repa_encoder
