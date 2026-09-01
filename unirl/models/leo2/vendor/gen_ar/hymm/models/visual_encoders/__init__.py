from pathlib import Path

from ...constants import VISION_ENCODER_META_INFO
from ...utils.torch_utils import PRECISION_TO_TYPE


def load_vision_model(
        vision_model_type=None,
        vision_model_precision=None,
        device=None,
        logger=None,
        require_grad=False,
        eval_mode=True,
        no_load_pretrained=False,
        vision_model_params=None,
        config=None,
):
    if logger is None:
        from loguru import logger

    if config is not None:
        vision_model_type = config["vision_model_type"]
        vision_model_precision = config.get("vision_model_precision", vision_model_precision)
        require_grad = not config.get("vision_model_freeze", not require_grad)
        eval_mode = config.get("vision_model_freeze", eval_mode)
        no_load_pretrained = config.get("no_load_pretrained_vision_model", False)
        vision_model_params = config.get("vision_model_params", vision_model_params)

    if vision_model_params is None:
        vision_model_params = {}

    if "janus-siglip" in vision_model_type:
        # No need to load ckpt, since it serves as a sub-model of janus-series models
        from .janus_siglip import create_siglip_vit, SigLIP_MODEL_CONFIG

        janus_siglip_encoder_name = vision_model_type.split("-")[1]
        assert janus_siglip_encoder_name in SigLIP_MODEL_CONFIG, f"Vision model type {janus_siglip_encoder_name} not found in SigLIP_MODEL_CONFIG"
        vision_model = create_siglip_vit(janus_siglip_encoder_name)

    elif vision_model_type == "siglip2-so400m-patch16-naflex":
        from .siglip2.model import Siglip2VisionTransformer

        vision_model_meta_info = VISION_ENCODER_META_INFO[vision_model_type]
        vision_model_path = Path(vision_model_meta_info["path"])
        vision_model = Siglip2VisionTransformer.from_pretrained(
            vision_model_path, pretrained=not no_load_pretrained, **vision_model_params)

    elif vision_model_type == "siglip2-large-patch16-512":
        from .siglip.modeling_siglip import SiglipVisionModel

        if no_load_pretrained:
            raise NotImplementedError(f"no_load_pretrained is not implemented for {vision_model_type}.")

        vision_model_meta_info = VISION_ENCODER_META_INFO[vision_model_type]
        vision_model_path = Path(vision_model_meta_info["path"])
        vision_model = SiglipVisionModel.from_pretrained(vision_model_path)

    elif vision_model_type.startswith("anyres-vit"):
        from .anyres.anyres_vit import AnyResViT, AnyResViTConfig

        model_config = AnyResViTConfig.from_name(vision_model_type)
        vision_model = AnyResViT(model_config, **vision_model_params)
    elif vision_model_type.startswith("qwen3vl-vit"):
        from .qwen.qwen_vit import Qwen3VLVisionModel, QwenViTConfig
        model_config = QwenViTConfig.from_name(vision_model_type)
        vision_model = Qwen3VLVisionModel(model_config, **vision_model_params)
    else:
        raise NotImplementedError(f"vision_model_type {vision_model_type} not implemented")

    if vision_model_precision is not None:
        logger.warning(f"You are transforming the Vision Encoder to {vision_model_precision}. Please make sure this is what you want.")
        if isinstance(vision_model_precision, str):
            vision_model_precision = PRECISION_TO_TYPE[vision_model_precision]
        vision_model = vision_model.to(dtype=vision_model_precision)

    if device is not None:
        vision_model = vision_model.to(device=device)

    if not require_grad:
        vision_model.requires_grad_(False)

    if eval_mode:
        vision_model.eval()

    return vision_model


def load_vision_model_processor(vision_model_type, **kwargs):

    if vision_model_type == "siglip2-large-patch16-512":
        from transformers import SiglipImageProcessor
        vision_model_meta_info = VISION_ENCODER_META_INFO[vision_model_type]
        vision_model_path = Path(vision_model_meta_info["path"])
        processor = SiglipImageProcessor.from_pretrained(vision_model_path, **kwargs)

    elif vision_model_type == "siglip2-so400m-patch16-naflex":
        from transformers import Siglip2ImageProcessorFast
        vision_model_meta_info = VISION_ENCODER_META_INFO[vision_model_type]
        vision_model_path = Path(vision_model_meta_info["path"])
        processor = Siglip2ImageProcessorFast.from_pretrained(vision_model_path, **kwargs)

    elif vision_model_type.startswith("anyres-vit"):
        from .anyres.anyres_vit import AnyResViTImageProcessor, AnyResViTConfig
        model_config = AnyResViTConfig.from_name(vision_model_type)
        processor = AnyResViTImageProcessor(model_config, **kwargs)
    # use original image processor from transformers for QWenVL series
    elif vision_model_type.startswith("qwen3vl-vit"):
        from transformers import AutoImageProcessor
        vision_model_meta_info = VISION_ENCODER_META_INFO[vision_model_type]
        vision_model_path = Path(vision_model_meta_info["path"])
        processor = AutoImageProcessor.from_pretrained(vision_model_path, **kwargs)
    else:
        raise NotImplementedError(f"vision_model_type {vision_model_type} not implemented")

    return processor


def load_vision_model_video_processor(vision_model_type, **kwargs):
    """Load the native video processor paired with a vision encoder.

    Unlike the image processor, this one consumes a whole clip and folds every
    `temporal_patch_size` frames into one tubelet, so the encoder sees real inter-frame motion.
    """

    if vision_model_type.startswith("qwen3vl-vit"):
        from transformers import AutoVideoProcessor

        vision_model_meta_info = VISION_ENCODER_META_INFO[vision_model_type]
        vision_model_path = Path(vision_model_meta_info["path"])
        return AutoVideoProcessor.from_pretrained(vision_model_path, **kwargs)
    raise NotImplementedError(
        f"Video processor for vision_model_type {vision_model_type} is not implemented"
    )


load_vit = load_vision_model
load_vit_processor = load_vision_model_processor
load_vit_video_processor = load_vision_model_video_processor


def load_insight_face(insightface_path=None):
    from insightface.app import FaceAnalysis

    if insightface_path is None:
        from ...constants import VISION_ENCODER_META_INFO

        insightface_path = VISION_ENCODER_META_INFO["insightface"]["path"]

    name = 'buffalo_l'  # From large to small, support antelopev2, buffalo_l, buffalo_sc
    allowed_modules = ['detection', "recognition"]
    face_analysis = FaceAnalysis(name=name, root=insightface_path, allowed_modules=allowed_modules,
                                 providers=['CPUExecutionProvider'])
    face_analysis.prepare(ctx_id=0, det_size=(640, 640))
    return face_analysis

# for training vision encoder
def build_model(args, logger, rank, world_size, dtype=None, device=None):
    from .siglip2.model import SIGLIP2_CONFIG, Siglip2
    from hymm.utils.helpers import merge_dicts
    from easydict import EasyDict

    factor_kwargs = {"device": device, "dtype": dtype}
    if args.model_name in SIGLIP2_CONFIG.keys():
        config = SIGLIP2_CONFIG[args.model_name]
        config = merge_dicts(config, args.model_kwargs)
        config = EasyDict(config)
        model = Siglip2(config, rank=rank, world_size=world_size)
        model = model.to(**factor_kwargs)
        return model, config
    else:
        raise NotImplementedError(f"vision_model_type {args.model_name} not implemented")
