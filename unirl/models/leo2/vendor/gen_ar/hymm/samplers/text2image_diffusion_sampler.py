from hymm.constants import LI_DIT_PROMPT_TEMPLATE
from hymm.core.extra_model_provider import build_vae
from hymm.core.global_vars import get_args, get_logger
from hymm.models.text_encoder import TextEncoder
from hymm.samplers.hunyuan_multimodal_sampler import HunyuanMultimodalSampler


class Text2ImageDiffusionSampler(HunyuanMultimodalSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Manage arguments
        args = get_args()
        args.bot_task = "image"
        if args.image_size == "auto":
            args.image_size = "1:1"

    def setup_extra_models(self):
        args = get_args()
        logger = get_logger()

        # Initialize vae, text encoder
        self.model.vae = build_vae(dp_rank=self.rank, only_encoder=False)
        self.model.text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder_type,
            max_length=args.text_encoder_text_len,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer_name,
            use_attention_mask=args.text_encoder_use_attention_mask,
            infer_mode=args.text_encoder_infer_mode,
            prompt_template=(
                LI_DIT_PROMPT_TEMPLATE[args.text_encoder_prompt_template]
                if args.text_encoder_prompt_template is not None else None
            ),
            hidden_state_skip_layer=args.text_encoder_hidden_state_skip_layer,
            apply_final_norm=args.text_encoder_apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device=self.device,
            attn_implementation=getattr(args, "text_encoder_attn", None),
        )
