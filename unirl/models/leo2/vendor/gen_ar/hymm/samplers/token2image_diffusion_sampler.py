import torch

from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.config import parse_eval_initial_args
from hymm.samplers.base_sampler import BaseSampler
from hymm.models import load_vae
from hymm.models.visual_encoders import load_vision_model
from hymm.models.diffusion.posemb_layers import get_nd_rotary_pos_embed
from hymm.utils.file_utils import rank0_logger
from hymm.diffusion import load_diffusion_pipeline
from hymm.constants import VISION_ENCODER_META_INFO

from transformers.models.siglip2.image_processing_siglip2_fast import Siglip2ImageProcessorFast


class Token2ImageDiffusionSampler(BaseSampler):
    def __init__(self, args, model_dict, ckpt_path=None, rank=0, world_size=1, device=0, logger=None):
        super().__init__(
            args=args,
            model_dict=model_dict,
            ckpt_path=ckpt_path,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )

        pipeline_name = "token2image"
        self.pipeline = load_diffusion_pipeline(
            args=self.args,
            rank=self.rank,
            pipeline_name=pipeline_name,
            diffusion_model=self.model_dict["model"],
            vae=self.model_dict["vae"],
            device=self.device,
        )

        self.vae_downsample_factor = self.model_dict["vae"]._downsample_factor

    @staticmethod
    def build_extra_model(args, model_dict, factor_kwargs, logger=None):
        device = factor_kwargs["device"]
        vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=device,
            logger=logger,
        )
        model_dict["vae"] = vae

        # vision_encoder_processor = Siglip2ImageProcessorFast.from_pretrained(VISION_ENCODER_META_INFO[args.vision_encoder_type]["path"])
        # vision_encoder_max_num_patches = args.vision_encoder_max_num_patches
        # vision_encoder = load_vision_model(
        #     args.vision_encoder_type,
        #     args.vision_encoder_precision,
        #     device=device,
        #     logger=logger,
        # )
        # model_dict["vision_encoder"] = vision_encoder
        # model_dict["vision_encoder_processor"] = vision_encoder_processor
        # model_dict["vision_encoder_max_num_patches"] = vision_encoder_max_num_patches

        return model_dict
    
    # condition: image tensor  batch_size x channels x height x width
    # condition_embeds: siglip feature sequence  batch_size x seq_len x feature_dim
    # mutual exclusive, must provide one of condition or condition_embeds
    @torch.no_grad()
    def predict(self, args, image_size, **kwargs):

        token_height = image_size[0] // (self.vae_downsample_factor[0] * self.args.patch_size)
        token_width = image_size[1] // (self.vae_downsample_factor[1] * self.args.patch_size)

        target_ndim = 2     # n-d RoPE

        rope_sizes = [token_height, token_width]
        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model_dict["model_settings"].hidden_size // self.model_dict["model_settings"].num_heads
        rope_dim_list = self.model_dict["model_settings"].rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=self.model_dict["model_settings"].rope_theta,
            use_real=True,
            theta_rescale_factor=1.0,
        )

        image = self.pipeline(
            sample_size=(token_height, token_width),
            num_inference_steps=args.diff_infer_steps,
            guidance_scale=args.guidance_scale,
            model_input_extra_kwargs={
                "cond": torch.randn(1, 1024, 1152).cuda(),
                "cond_mask": torch.ones(1, 1024, dtype=torch.bool).cuda(),
                "freqs_cos": freqs_cos,
                "freqs_sin": freqs_sin,
            }
        )


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)

    sampler = Token2ImageDiffusionSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args

    # Start evaluation
    sampler.predict(args=args, image_size=(512, 512))



if __name__ == "__main__":
    main()
