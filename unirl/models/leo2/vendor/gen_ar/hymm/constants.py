import os
from pathlib import Path

BASE_CONFIG_PATH = Path(os.getenv("BASE_CONFIG_PATH", str(Path(__file__).parent / "configs/base.yaml")))

PRECISIONS = {"fp32", "fp16", "bf16"}
SHARDING_STRATEGIES = {"FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"}

# ===== launcher =====
LAUNCHER = "deepspeed"

def set_launcher(launcher):
    global LAUNCHER
    LAUNCHER = launcher


# =================== Constant Values =====================
# Computation scale factor, 1P = 1_000_000_000_000_000. Tensorboard will display the value in PetaFLOPS to avoid
# overflow error when tensorboard logging values.
C_SCALE = 1_000_000_000_000_000

# ================ Data Loader ================
IMAGE_CROP_TYPE = {"random", "center"}
INDEX_STRATEGY = {"uniform", "probability"}

# ================ Visual Diffusion ================
# Closed set of supported channel-concat extension types for the diffusion model.
SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES = ("t2i", "t2a", "t2v", "i2v", "fl2v")

# ================ Module Assets (training and evaluation) ================
ASSETS_BASE = os.getenv("ASSETS_BASE", "").rstrip('/')

VAE_BASE = os.getenv("VAE_BASE", f"{ASSETS_BASE}/image_encoder").rstrip('/')
AUDIO_VAE_BASE = os.getenv("VAE_BASE", f"{ASSETS_BASE}/audio_encoder").rstrip('/')
VAE_META_INFO = {
    # ============================================
    # =              Image Tokenizer             =
    # ============================================
    "88-magvitv2-hy_240930": {  # fp32
        "path": f"{VAE_BASE}/magvit_bsq_18c_2d_hy/240930",
        "codebook_size": 262144,
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "88-vqgan-hy_241017": {  # fp32
        "path": f"{VAE_BASE}/vqgan_2d_hy/241017",
        "codebook_size": 16384,
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "88-vqgan-hy_241024": {  # fp32
        "path": f"{VAE_BASE}/vqgan_2d_hy/241024",
        "codebook_size": 16384,
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "88-vqgan-hy_241202": {  # fp32
        "path": f"{VAE_BASE}/vqgan_2d_hy/241202",
        "codebook_size": 16384,
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "1616-vqgan-hy_241223": {  # fp32
        "path": f"{VAE_BASE}/vqgan_2d_hy/241223",
        "codebook_size": 16384,
        "downsample_factor": [16, 16],
        "trans_type": "-11",
    },
    "88-magvitv2-show-o": {  # fp32
        "path": f"{VAE_BASE}/magvitv2_show_o",
        "codebook_size": 8192,
        "downsample_factor": [16, 16],
        "trans_type": "-11",
    },
    "88-vqgan-sd": {  # fp32
        "path": f"{VAE_BASE}/vq-f8-sd",
        "codebook_size": 16384,
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "88-movqgan-vanilla": {  # fp32
        "path": f"{VAE_BASE}/MoVQGAN",
        "codebook_size": 16384,
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "88-movqgan-emu3": {  # fp32
        "path": f"{VAE_BASE}/Emu3_VisionTokenizer",
        "codebook_size": 32768,
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "88-vqgan-maskgit": {  # fp32
        "path": f"{VAE_BASE}/maskgit-vqvae",
        "codebook_size": 1024,
        "downsample_factor": [8, 8],
        "trans_type": "01",
    },
    "1616-vqgan-maskgit_pytorch":{
        "path": f"{VAE_BASE}/maskgit-vqvae/maskgit-pytorch",
        "codebook_size": 1024,
        "downsample_factor": [16, 16],
        "trans_type": "01",
    },
    "1616-vq-janus": {  # bf16
        "path": f"{VAE_BASE}/vq_januspro",
        "codebook_size": 16384,
        "downsample_factor": [16, 16],
        "trans_type": "-11",
    },

    # ============================================
    # =         Image Semantic Encoder           =
    # ============================================
    "1414-evaclip-emu2": {  # bf16
        "path": f"{VAE_BASE}/evaclip_emu2",
        "downsample_factor": [14, 14],
        "trans_type": "-11",
    },
    "32x32-evaclip-sdxl": {
        "path": f"{VAE_BASE}/evaclip_256x16x16_sdxl",
        "downsample_factor": [64, 64],  # 1024 -> 16
        "trans_type": "-11",
        # the default range of evaclip-sdxl is [0, 1], can be set to [-1, 1], the most common case is [-1, 1]
        "return_dict": True,
    },

    # ============================================
    # =                 Image VAE                =
    # ============================================
    "88-vae-sdxl": {  # bf16
        "path": f"{VAE_BASE}/vae_f8_sdxl",
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "88-4c-sdxl": {  # bf16
        "path": f"{VAE_BASE}/vae_f8_sdxl",
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "88-16c-hy": {  # bf16
        "path": f"{VAE_BASE}/vae_2d/hyvae_v1_0723",
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    "88-16c-flux1": {
        "path": f"{VAE_BASE}/flux-vae",
        "downsample_factor": [8, 8],
        "trans_type": "-11",
    },
    # 实际是8x8-32c，但是在vae encode后做了2x2的patchify，实际等效16x16-128c
    # 在vae外不需要手动scale shift，encode函数自带将结果normalize，decode函数自带将输入unnormalize
    "16x16-128c-flux2": {
        "path": f"{VAE_BASE}/flux2-vae",
        "downsample_factor": [16, 16],
        "trans_type": "-11",
    },
    "32x32-64c-hy":{
        "path": f"{VAE_BASE}/vae_2d/hyvae_32x32_64c_v1_0115",
        "downsample_factor": [32, 32],
        "trans_type": "-11",
    },
    "32x32-64c-hy_v2":{
        "path": f"{VAE_BASE}/vae_2d/hyvae_32x32_64c_v2_0124",
        "downsample_factor": [32, 32],
        "trans_type": "-11",
    },
    "16x16x4-32c-hy":{
        "path": f"{VAE_BASE}/vae_3d/hyvae_16x16x4_32c",
        "downsample_factor": [16, 16],  # we only use 2D for image generation
        "trans_type": "-11",
        "class_type": "HYVAE3D",
    },
    "16x16x4-32c-hy-image":{
        "path": f"{VAE_BASE}/vae_3d/hyvae_16x16x4_32c_image",
        "downsample_factor": [16, 16],
        "trans_type": "-11",
        "class_type": "HYVAE3D",
    },

    # ============================================
    # =                Video VAE                 =
    # ============================================
    "16x16x4-32c-hy-20250605": {
        "path": f"{VAE_BASE}/vae_3d/hyvae_f16x16x4_c32_20250605",
        "downsample_factor": [16, 16],
        "duration_downsample_factor": 4,
        "trans_type": "-11",
        "class_type": "HYVAE3D_RMSNorm",
        "latent_dim": 32,
    },
    "16x16x4-48c-hy-v3": {
        "path": f"{VAE_BASE}/vae_3d/hyvae_vid_leo2.0_v2.3.0",
        "downsample_factor": [16, 16],
        "duration_downsample_factor": 4,
        "trans_type": "-11",
        "class_type": "HYVAE3D_RMSNorm_v3",
        "latent_dim": 48,
    },
    "16x16x4-48c-hy-v3_3": {
        "path": f"{VAE_BASE}/vae_3d/hyvae_vid_leo2.0_v2.5.0",
        "downsample_factor": [16, 16],
        "duration_downsample_factor": 4,
        "trans_type": "-11",
        "class_type": "HYVAE3D_RMSNorm_v3_3",
        "latent_dim": 48,
    },
    "16x16x4-48c-hy-v3_3-release": {
        "path": f"{VAE_BASE}/vae_3d/hyvae_vid_leo2.0_v2.5.1",
        "downsample_factor": [16, 16],
        "duration_downsample_factor": 4,
        "trans_type": "-11",
        "class_type": "HYVAE3D_RMSNorm_v3_3",
        "latent_dim": 48,
    },
    "16x16x4-48c-hy-v3_3-release2": {
        "path": f"{VAE_BASE}/vae_3d/hyvae_vid_leo2.0_v2.5.1_release2",
        "downsample_factor": [16, 16],
        "duration_downsample_factor": 4,
        "trans_type": "-11",
        "class_type": "HYVAE3D_RMSNorm_v3_3",
        "latent_dim": 48,
    },    
    "16x16-64c-hy3.5":{
        "path": f"{VAE_BASE}/vae_2d/hyvae_16x16_64c_3_5",
        "downsample_factor": [16, 16],
        "trans_type": "-11",
    },

    # ============================================
    # =                Audio VAE                 =
    # ============================================
    "dac-24khz": {
        "path": f"{VAE_BASE}/dac_24khz_single_4096",
        "downsample_factor": [8, 8],
        "codebook_size": 4096,
    },
    "dac-24khz-double": {
        "path": f"{AUDIO_VAE_BASE}/dac_24khz_double_8192",
        "downsample_factor": [8, 8],
        "codebook_size": 16384,
    },
    "dac-24khz-double-single": {
        "path": f"{AUDIO_VAE_BASE}/dac_24khz_double_8192",
        "downsample_factor": [8, 8],
        "codebook_size": 8192,
    },
}

VISION_ENCODER_BASE = os.getenv("VISION_ENCODER_BASE", f"{ASSETS_BASE}/vision_encoder").rstrip('/')
VISION_ENCODER_META_INFO = {
    "siglip-large-patch16-384": {
        "path": f"{VISION_ENCODER_BASE}/siglip-large-patch16-384",
        "downsample_factor": [16, 16],
        "image_size": 384,
        "dummy_number": 1,
    },
    "siglip2-large-patch16-512": {
        "path": f"{VISION_ENCODER_BASE}/siglip2-large-patch16-512",
        "downsample_factor": [16, 16],
        "image_size": 512,
        "dummy_number": 1,
    },
    "siglip2-so400m-patch16-naflex": {
        "path": f"{VISION_ENCODER_BASE}/siglip2-so400m-patch16-naflex",
        "downsample_factor": [16, 16],
        "dummy_number": 1,
    },
    "insightface": {
        "path": f"{VISION_ENCODER_BASE}/insightface",
    },
    "anyres-vit-for-a3b": {
        "downsample_factor": [32, 32],
        "dummy_number": 4,  # 1(dummy) + 1(new_line) + 2(begin, end)
    },
    "anyres-vit-for-hy3-a3b": {
        "downsample_factor": [32, 32],
        "dummy_number": 2,  # 1(dummy) + 1(new_line)
        "cat_extra_token": False,
    },
    "anyres-vit-for-a30b": {
        "path": f"{VISION_ENCODER_BASE}/anyres-vit-hunyuan-moe-a30b-standalone",
        "downsample_factor": [32, 32],
        "dummy_number": 4,  # 1(dummy) + 1(new_line) + 2(begin, end)
    },
    "qwen3vl-vit-for-30b-a3b": {
        "path": f"{VISION_ENCODER_BASE}/Qwen3-VL-30B-A3B-Instruct",
        "downsample_factor": [32, 32],
        "spatial_merge_size": 2,
        "patch_dim": 1536,
        "dummy_number": 4,  # 1(dummy) + 1(new_line) + 2(begin, end)
    },
    "qwen3vl-vit-for-qwen3.5-9b": {
        "path": f"{VISION_ENCODER_BASE}/Qwen3.5-9B",
        "downsample_factor": [32, 32],
        "spatial_merge_size": 2,
        "patch_dim": 1152,
        "dummy_number": 4,  # vit merger uses 2x2 tokens
    },
    "DINOv3": {
        "path": f"{VISION_ENCODER_BASE}/DINOv3",
    },
    "qwen-3-vl-8b-instruct": {
        "path": f"{VISION_ENCODER_BASE}/Qwen3-VL-8B-Instruct",
    },
    "qwen-3.5-9b": {
        "path": f"{VISION_ENCODER_BASE}/Qwen3.5-9B",
    },
}


AUDIO_ENCODER_BASE = os.getenv("AUDIO_ENCODER_BASE", f"{ASSETS_BASE}/audio_encoder").rstrip('/')
AUDIO_ENCODER_META_INFO = {
    "waveflow-v1_0": {
        "path": f"{AUDIO_ENCODER_BASE}/waveflow_v1_0/g_00640000",
        "stats": f"{AUDIO_ENCODER_BASE}/waveflow_v1_0/global_mean_var_64w.stat",
        "channels": 1,
        "sample_rate": 24000,
    },
    "dual_channel_48k": {
        "path": f"{AUDIO_ENCODER_BASE}/dual_channel_48k/vae_audio_192d96l_3087k.ckpt",
        "config": f"{AUDIO_ENCODER_BASE}/dual_channel_48k/stable_audio_1920_vae_htae_32gpu.json",
        "mean_std": f"{AUDIO_ENCODER_BASE}/dual_channel_48k/vae_audio_192d96l_3087k_mean_std",
        "channels": 2,
        "sample_rate": 48000,
        "downsampling_ratio": 1920,
    },
    "dual_channel_48k_refine_decoder": {
        "path": f"{AUDIO_ENCODER_BASE}/dual_channel_48k_refine_decoder/vae_audio_192d96l_3087k.ckpt",
        "config": f"{AUDIO_ENCODER_BASE}/dual_channel_48k_refine_decoder/stable_audio_1920_vae_htae_32gpu.json",
        "mean_std": f"{AUDIO_ENCODER_BASE}/dual_channel_48k_refine_decoder/vae_audio_192d96l_3087k_mean_std",
        "channels": 2,
        "sample_rate": 48000,
        "downsampling_ratio": 1920,
    },
}


TEXT_ENCODER_BASE = os.getenv("TEXT_ENCODER_BASE", f"{ASSETS_BASE}/text_encoder").rstrip('/')
# ================ Text Encoder ================
# When using decoder-only models, we must provide a prompt template to instruct the text encoder
# on how to generate the text features.
# --------------------------------------------------------------------
LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_GENERATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>"
)
LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)   # Add an <|eot_id|> at the end of the prompt to avoid the full-zero attention mask when prompt is empty.
# 128000,
# 128006, 9125, 128007, 271, 75885, 279, 2217, 555, 45293, 279, 1933, 11, 6211, 11, 1404, 11, 10651, 11,
# 12472, 11, 1495, 11, 29079, 12135, 315, 279, 6302, 323, 4092, 25, 128009,
# 128006, 882, 128007, 271,
# --------------------------------------------------------------------
GLM_4V_9B_PROMPT_TEMPLATE_LI_DIT_ENCODE = [
    {"role": "system", "content": "Describe the image by detailing the color, shape, size, texture, "
                                  "quantity, text, spatial relationships of the objects and background:"},
    {"role": "user", "content": "{}"},
]
# 151331, 151333, 151335, 198, 74198, 279, 2168, 553, 43937, 279, 1894, 11, 6083, 11, 1379, 11, 10429, 11,
# 12188, 11, 1467, 11, 27884, 11865, 315, 279, 6171, 323, 4004, 25, 151336,
# --------------------------------------------------------------------
QWEN_2_5_VL_32B_INSTRUCT_PROMPT_TEMPLATE_LI_DIT_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}"
)
QWEN_2_5_VL_72B_INSTRUCT_PROMPT_TEMPLATE_LI_DIT_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}"
)
# --------------------------------------------------------------------
# 151644, 8948, 198, 74785, 279, 2168, 553, 44193, 279, 1894, 11, 6083, 11, 1379, 11, 10434, 11, 12194, 11, 1467, 11, 27979, 11871, 315, 279, 6171, 323, 4004, 25, 151645, 198, 151644, 872, 198, {}, 151645, 198, 151644, 77091, 198


LI_DIT_PROMPT_TEMPLATE = {
    "li-dit-generate": {"template": LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_GENERATE},
    "li-dit-encode": {"template": LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_ENCODE, "crop_start": 36},
    "li-dit-encode-llava-3": {"template": LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_ENCODE, "crop_start": 36},
    "li-dit-encode-glm-4v": {"template": GLM_4V_9B_PROMPT_TEMPLATE_LI_DIT_ENCODE, "crop_start": 31},
    "li-dit-encode-qwen-2.5-vl-32b-instruct": {"template": QWEN_2_5_VL_32B_INSTRUCT_PROMPT_TEMPLATE_LI_DIT_ENCODE, "crop_start": 34},
    "li-dit-encode-qwen-2.5-vl-72b-instruct": {"template": QWEN_2_5_VL_72B_INSTRUCT_PROMPT_TEMPLATE_LI_DIT_ENCODE, "crop_start": 34},
}

TEXT_ENCODER_PATH = {
    "t5": f"{TEXT_ENCODER_BASE}/PixArt-XL-2-512x512/text_encoder",
    "t5_v11_xxl": f"{TEXT_ENCODER_BASE}/t5_v1_1_xxl",
    "clipL": f"{TEXT_ENCODER_BASE}/openai_clip-vit-large-patch14",
    "llava-llama-3-8b": f"{TEXT_ENCODER_BASE}/llava-llama-3-8b-v1_1-pure-llama",
    "glm-4v-9b": f"{TEXT_ENCODER_BASE}/glm-4v-9b-pure-glm",
    "qwen-2.5-vl-32b-instruct": f"{TEXT_ENCODER_BASE}/qwen_2_5_vl_32b_instruct",
    "qwen-2.5-vl-72b-instruct": f"{TEXT_ENCODER_BASE}/qwen_2_5_vl_72b_instruct",
    "qwen-2.5-vl-7b-instruct": f"{TEXT_ENCODER_BASE}/Qwen2.5-VL-7B-Instruct",
    "qwen-3-vl-8b-instruct": f"{TEXT_ENCODER_BASE}/Qwen3-VL-8B-Instruct",
    "qwen-3vl-8b": f"{TEXT_ENCODER_BASE}/Qwen3-VL-8B-Instruct",
    "qwen-3.5-9b": f"{TEXT_ENCODER_BASE}/Qwen3.5-9B",
    "qwen-3.5-35-a3b": f"{TEXT_ENCODER_BASE}/Qwen3.5-35B-A3B",
    "qwen-3-omni-30-a3b": f"{TEXT_ENCODER_BASE}/Qwen3-Omni-30B-A3B-Instruct"
}
TEXT_ENCODER_TOKENIZER_PATH = {
    "t5": f"{TEXT_ENCODER_BASE}/PixArt-XL-2-512x512/tokenizer",
    # If the tokenizer path is not specified, we will use the same path as the text encoder path
}


PRETRAINED_LLM_BASE = os.getenv("PRETRAINED_BASE", f"{ASSETS_BASE}/pretrained_llm").rstrip('/')
PRETRAINED_LLM_PATH = {
    "phi-2": f"{PRETRAINED_LLM_BASE}/phi-2/phi-2.pt",
    "phi-2-interleave": f"{PRETRAINED_LLM_BASE}/phi-2/phi-2-interleave.pt",
    "hunyuan-dense-3b": f"{PRETRAINED_LLM_BASE}/Hunyuan3B-Dense-SFT-32k/model.pt", # 这是一个SFT的版本
    "hunyuan-dense-3b-interleave": f"{PRETRAINED_LLM_BASE}/Hunyuan3B-Dense-SFT-32k/model-interleave.pt", # 这是一个SFT的版本
    "hunyuan-dense-7b": f"{PRETRAINED_LLM_BASE}/Hunyuan7B-Dense-Pretrain/hf_pretrain_256k_v2_241208/transformed_model.pt", # 这是一个Pretrain的版本
    "hunyuan-dense-70b": f"{PRETRAINED_LLM_BASE}/Hunyuan70B-Dense-Pretrain/hf_pretrain_32k_v2_250328/transformed_model.pt", # 这是一个Pretrain的版本
    "hunyuan-moe-7b-a13b": f"{PRETRAINED_LLM_BASE}/Hunyuan7B-MOE-32K-pretrain/transformed_model.pt", # 这是一个Pretrain的版本
    "hunyuan-moe-7b-a13b-vocab-expanded": f"{PRETRAINED_LLM_BASE}/Hunyuan7B-MOE-32K-pretrain/transformed_model.pt", # 这是一个Pretrain的版本
    "qwen-2.5-7b": f"{PRETRAINED_LLM_BASE}/Qwen2.5-7B/transformed_model.pt", # 这是一个Pretrain的版本
    "qwen-2.5-7b-instruct": f"{PRETRAINED_LLM_BASE}/Qwen2.5-7B-Instruct/transformed_model.pt", # 这是一个SFT的版本
    "qwen-2.5-vl-7b-instruct": f"{PRETRAINED_LLM_BASE}/Qwen2.5-VL-7B-Instruct/transformed_model.pt", # 这是一个SFT的版本
    "deepseek-r1-distill-qwen-7b": f"{PRETRAINED_LLM_BASE}/DeepSeek-R1-Distill-Qwen-7B/transformed_model.pt", # 这是一个DeepSeek-R1蒸馏的版本
    "deepseek-llm-7b-base": f"{PRETRAINED_LLM_BASE}/DeepSeekLLM-7B-base/transformed_model.pt", # 这是一个Pretrain的版本
    "deepseek-llm-7b-chat": f"{PRETRAINED_LLM_BASE}/DeepSeekLLM-7B-chat/transformed_model.pt", # 这是一个SFT的版本
    "qwen3-vl-30b-a3b-instruct": f"{PRETRAINED_LLM_BASE}/Qwen3-VL-30B-A3B-Instruct-extended",
}

WTE_LN_F_PATH = {
    "hunyuan-dense-7b": f"{PRETRAINED_LLM_BASE}/Hunyuan7B-Dense-Pretrain/hf_pretrain_256k_v2_241208/wte_ln_f.pt", # 这是一个Pretrain的版本
    "hunyuan-dense-70b": f"{PRETRAINED_LLM_BASE}/Hunyuan70B-Dense-Pretrain/hf_pretrain_32k_v2_250328/wte_ln_f.pt", # 这是一个Pretrain的版本
    "hunyuan-moe-7b-a13b": f"{PRETRAINED_LLM_BASE}/Hunyuan7B-MOE-32K-pretrain/wte_ln_f.pt", # 这是一个Pretrain的版本
    "hunyuan-moe-7b-a13b-vocab-expanded": f"{PRETRAINED_LLM_BASE}/Hunyuan7B-MOE-32K-pretrain/wte_ln_f_expanded.pt", # 这是一个Pretrain的版本
}

TOKENIZER_BASE = os.getenv("TOKENIZER_BASE", f"{ASSETS_BASE}/pretrained_llm").rstrip('/')
TOKENIZER_PATH = {
    "phi-2": f"{TOKENIZER_BASE}/phi-2",
    "hunyuan-dense-3b": f"{TOKENIZER_BASE}/Hunyuan-common/inference/OpenSourceTokenizerSft3B",
    "hunyuan-dense-7b": f"{TOKENIZER_BASE}/Hunyuan7B-Dense-Pretrain/hf_pretrain_256k_v2_241208",
    "hunyuan-dense-7b-tokenizer-refactor": f"{TOKENIZER_BASE}/Hunyuan7B-Dense-Pretrain/hf_pretrain_256k_v2_241208_tokenizer_refactor",
    "hunyuan-dense-70b": f"{TOKENIZER_BASE}/Hunyuan7B-Dense-Pretrain/hf_pretrain_256k_v2_241208_tokenizer_refactor",
    "hunyuan-moe-7b-a13b": f"{TOKENIZER_BASE}/Hunyuan7B-Dense-Pretrain/hf_pretrain_256k_v2_241208_tokenizer_refactor",
    "qwen-2.5-7b": f"{TOKENIZER_BASE}/Qwen2.5-7B",
    "qwen-2.5-7b-instruct": f"{TOKENIZER_BASE}/Qwen2.5-7B-Instruct",
    "qwen-2.5-vl-7b-instruct": f"{TOKENIZER_BASE}/Qwen2.5-VL-7B-Instruct",
    "deepseek-r1-distill-qwen-7b": f"{TOKENIZER_BASE}/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-llm-7b-base": f"{TOKENIZER_BASE}/DeepSeekLLM-7B-base",
    "deepseek-llm-7b-chat": f"{TOKENIZER_BASE}/DeepSeekLLM-7B-chat",
    "janus-pro-7b": f"{TOKENIZER_BASE}/Janus-Pro-7B",
    "hunyuan-moe-a3b": f"{TOKENIZER_BASE}/HunyuanMoE-A3B-MidTrain",
    "hunyuan-moe-a3b-multimodal-v2": f"{TOKENIZER_BASE}/extended_tokenizer/hunyuan_multimodal_v2",
    "hunyuan-moe-a3b-multimodal-v3": f"{TOKENIZER_BASE}/extended_tokenizer/hunyuan_multimodal_v3",  # 256 ratio tokens
    "hunyuan-moe-a3b-multimodal-v4": f"{TOKENIZER_BASE}/extended_tokenizer/hunyuan_multimodal_v4",  # support video
    "llava-llama-3-8b": f"{TEXT_ENCODER_BASE}/llava-llama-3-8b-v1_1-pure-llama",
    "qwen3-vl-30b-a3b-instruct": f"{TOKENIZER_BASE}/Qwen3-VL-30B-A3B-Instruct-extended",            # extended with hunyuan tokenizer
    "hymm-v3-5": f"{TOKENIZER_BASE}/extended_tokenizer/hymm_v3_5",   # extended from HY Image 3.5
    "qwen-3-vl-8b-instruct": f"{TEXT_ENCODER_BASE}/Qwen3-VL-8B-Instruct",
    "qwen-3.5-9b": f"{TEXT_ENCODER_BASE}/Qwen3.5-9B",
    "qwen-3.5-35-a3b": f"{TEXT_ENCODER_BASE}/Qwen3.5-35B-A3B",
    "qwen-3-omni-30-a3b": f"{TEXT_ENCODER_BASE}/Qwen3-Omni-30B-A3B-Instruct",
    "hunyuan3-moe-a3b": f"{TOKENIZER_BASE}/v3_a3b_offical_ckpts/256k/hf",  # extended from HY3.0
    "hunyuan3-moe-a3b-hymm-v3-5": f"{TOKENIZER_BASE}/extended_tokenizer/hunyuan3_moe_a3b_hymm_v3_5",  # extended from HY3.0 for Hymm Image 3.5
}

# ----------------- Evaluation ckpt/data ----------------
EVAL_BASE = os.getenv("EVAL_BASE", f"{ASSETS_BASE}/evaluation").rstrip('/')

COCO3K_PATH = f"{EVAL_BASE}/dataset/COCO/COCO3k.json"
COCO6K_PATH = f"{EVAL_BASE}/dataset/COCO/COCO6k.json"
COCO30K_PATH = f"{EVAL_BASE}/dataset/COCO/COCO30k.json"

FID_INCEPTION_PATH = f"{EVAL_BASE}/FID/pt_inception-2015-12-05-6726825d.pth"
FID_TARGET_PATH = {
    "imagenet": f"{EVAL_BASE}/FID/imagenet_val_fid_stats.pt",
    "coco30k": f"{EVAL_BASE}/FID/coco_val_fid_stats.pt",
    "coco6k": f"{EVAL_BASE}/FID/coco_val_fid_stats.pt",
    "coco3k": f"{EVAL_BASE}/FID/coco_val_fid_stats.pt",
}

# CLIP Score: CLIP model
CLIP_MODEL_PATH = f"{EVAL_BASE}/CLIP/ViT-B-32.pt"
# HPSv2: HPSv2 model
HPSV2_MODEL_PATH = f"{EVAL_BASE}/xswu-HPSv2/HPS_v2.1_compressed.pt"
# T2I-CompBench: BLIP_VQA model
T2I_COMPBENCH_PATH = [
    "data/t2i_compbench/color_val.csv",
    "data/t2i_compbench/shape_val.csv",
    "data/t2i_compbench/texture_val.csv",
]
VQA_MODEL_PATH = f"{EVAL_BASE}/T2I-CompBench"
# DPG-Bench: MPLUG model
MPLUG_MODEL_PATH = f"{EVAL_BASE}/mplug"
DPG_BENCH_DATA = [
    "data/dpg_bench/data.csv",
    "data/dpg_bench/questions.json",
]
GENEVAL_MODEL_PATH = f"{EVAL_BASE}/GenEval"
GENEVAL_DATA = "data/geneval/geneval.jsonl"
# VLM Score
VLM_SCORE_DATA_PATH = f"{EVAL_BASE}/VLMScore"

# Counting-eval dataset
COUNTING_EVAL_DATA = "data/counting_eval/counting_eval.jsonl"

# MMLU-Bench dataset
MMLU_BENCH_DATA = f"{EVAL_BASE}/MMLU/data"

# MMLU-Pro-Bench dataset
MMLU_PRO_BENCH_DATA = f"{EVAL_BASE}/MMLU-Pro/data"

IMAGENET_VAL_PATH = f"{EVAL_BASE}/dataset/imagenet/imagenet_1k_val.json"

# CIDEr: Stanford CoreNLP
CIDER_TOKENIZER = f"{EVAL_BASE}/CIDEr/stanford-corenlp-3.4.1.jar"
CAPTION_TEST_PATH = {
    "flickr30k_test": f"{EVAL_BASE}/dataset/flickr30k/arrows/test/00000.arrow",
    "nocaps_val": f"{EVAL_BASE}/dataset/nocaps/arrows/val/00000.arrow",
    "mmu_general_qa": f"{EVAL_BASE}/CapsBench/general_qa/images",
    "mmu_ocr": f"{EVAL_BASE}/CapsBench/ocr/images",
    "mmu_ocr_det_book": f"{EVAL_BASE}/mmu_det/ocr_det_book/images",
    "mmu_ocrbench": f"{EVAL_BASE}/mmu_det/OCRBench",
}

# VQAv2
VQA_DATA_PATH = {
    'vqav2_val': {
        'train': f'{EVAL_BASE}/VQA/vqav2/vqav2_train.jsonl',
        'test': f'{EVAL_BASE}/VQA/vqav2/vqav2_val.jsonl',
        'question': f'{EVAL_BASE}/VQA/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': f'{EVAL_BASE}/VQA/vqav2/v2_mscoco_val2014_annotations.json',
        'max_new_tokens': 10,
    },
}
COCO_VAL2014 = f"{EVAL_BASE}/dataset/COCO/val2014"

# MMBench
LMUDataRoot = f"{EVAL_BASE}/MMBench"

# MMMU
MMMU_PATH = {
    "mmmu": {
        "path": f"{EVAL_BASE}/MMMU",
        "split": "validation",
    },
    "mmmu_pro": {
        "path": f"{EVAL_BASE}/MMMU_Pro",
        "split": "validation",
    }
}

# audio eval dataset
AUDIOSET = f"{EVAL_BASE}/dataset/audioset/val/"
VGGSOUND = f"{EVAL_BASE}/dataset/vggsound/val/"

# imagebind
IMAGEBIND_MODEL_PATH = f"{EVAL_BASE}/imagebind/imagebind_huge.pth"

# clap score
CLAP_MODEL_PATH = f"{EVAL_BASE}/larger_clap_general/"

# FAD
FAD_TARGET_PATH = {
    "audioset": f"{EVAL_BASE}/FAD/audioset_clap_large_fid_stats.pt",
    "vggsound": f"{EVAL_BASE}/FAD/vggsound_clap_large_fid_stats.pt",
}
FAD_VGGISH_TARGET_PATH = {
    "audioset": f"{EVAL_BASE}/FAD/audioset_vggish_fad_stats.pt",
    "vggsound": f"{EVAL_BASE}/FAD/vggsound_vggish_fad_stats.pt",
}

# FAD VGGISH
VGGISH_MODEL_PATH = f"{EVAL_BASE}/FAD/vggish-10086976.pth"
VGGISH_PCA_PATH = f"{EVAL_BASE}/FAD/vggish_pca_params.pth"

# Face Sim: weights of detection and recognition
REAL_WORLD_FACE_DATA_FOLDER = f"{EVAL_BASE}/dataset/real_world_face"
FACE_MODEL_PATH = f"{EVAL_BASE}/face"

# =================== Test constants =====================
TESTSET_TEMPLATE = os.environ.get("TESTSET_TEMPLATE", "data/test/{}.csv")


VALIDATION_METRICS = {}
LOSS_METRICS = {"val_loss"}
# fid, clip_score, hpsv2, t2i_compbench is for image
# fad, clap_score, imagebind_score is for audio
SCORE_METRICS = {"fid", "clip_score", "hpsv2", "t2i_compbench", "geneval", "fad", "clap_score", "imagebind_score"}
SAMPLE_METRICS = {"image", "audio"}


# ================ Denoise Schedule ================
# Flow Matching solvers
FLOW_SOLVER = {
    "euler",                # Euler solver
    "heun-2",               # Heun 2nd solver
    "midpoint-2",           # Midpoint 2nd solver
    "kutta-4",              # Runge-Kutta 4th solver
    "cfg++",                # CFG++
}

# ================ Predefined Key-Mappings ================
MODEL_KEY_MAPPING = {
    "hyimage3_hf_to_fsdp2": [
        'vision_model:vit',
        'vision_aligner:vit_aligner',
        'ln_f:norm',
        'wte:embed_tokens',
    ],
    "moe_a13b_ptm1_hf_to_fsdp2": [
        r'embed_tokens:__',
        r'language_model\.transformer\.ln_f:model.norm',
        r'language_model\.transformer\.wte:model.embed_tokens',
        r'language_model\.:',
        r'vision_aligner_so:vit_aligner',
        r'vision_model_so:vit|layernorm_mlp\.:',
    ],
    "moe_a13b_ptm2_to_fsdp2": ['|'.join([
        r'model\.layers:model.transformer.layers',
        r'model>self_attn:self_attention',
        r'model>qkv_proj:linear_qkv',
        r'model>o_proj:linear_proj',
        r'model>input_layernorm\.weight:self_attention.linear_qkv.layer_norm_weight',
        r'model>post_attention_layernorm:pre_mlp_layernorm',
        r'model>query_layernorm:q_layernorm',
        r'model>key_layernorm:k_layernorm',
        r'model>gate\.wg:router',
        r'model>expert_gate_and_up_weights:experts.experts.linear_fc1.weight',
        r'model>expert_down_weights:experts.experts.linear_fc2.weight',
        r'model>shared_mlp\.gate_and_up_proj:shared_experts.linear_fc1',
        r'model>shared_mlp\.down_proj:shared_experts.linear_fc2',
    ])],
    "moe_a3b_ptm2_to_fsdp2": ['|'.join([
        r'model\.layers:model.transformer.layers',
        r'model>self_attn:self_attention',
        r'model>qkv_proj:linear_qkv',
        r'model>o_proj:linear_proj',
        r'model>input_layernorm\.weight:self_attention.linear_qkv.layer_norm_weight',
        r'model>post_attention_layernorm:pre_mlp_layernorm',
        r'model>query_layernorm:q_layernorm',
        r'model>key_layernorm:k_layernorm',
        r'model>gate\.wg:router',
        r'model>expert_gate_and_up_weights:experts.experts.linear_fc1.weight',
        r'model>expert_down_weights:experts.experts.linear_fc2.weight',
        r'model>shared_mlp\.gate_and_up_proj:shared_experts.linear_fc1',
        r'model>shared_mlp\.down_proj:shared_experts.linear_fc2',
        r'model>\.0\.pre_mlp_layernorm\.weight:.0.mlp.linear_fc1.layer_norm_weight',
        r'model>mlp\.gate_and_up_proj:mlp.linear_fc1',
        r'model>mlp\.down_proj:mlp.linear_fc2',
    ])],
    "qwen3vl_30b_a3b_hf_to_fsdp2": [
        r'model\.language_model\.layers\.(\d+)\.self_attn\.q_norm:model.layers.\1.self_attn.query_layernorm',
        r'model\.language_model\.layers\.(\d+)\.self_attn\.k_norm:model.layers.\1.self_attn.key_layernorm',
        r'model\.visual:vit',
        r'language_model\.:',
    ],

    # For Leo2.0
    "leo2_8.5B_ptm2_to_fsdp2": [
        '|'.join([  # List order matters.
            r'layers:double_blocks.layers',
            r'mod_proj_txt:txt_mod',
            r'self_attn\.query_layernorm_txt:txt_attn_q_norm',
            r'self_attn\.key_layernorm_txt:txt_attn_k_norm',
            r'self_attn\.q_proj_txt:txt_attn_q',
            r'self_attn\.k_proj_txt:txt_attn_k',
            r'self_attn\.v_proj_txt:txt_attn_v',
            r'self_attn\.o_proj_txt:txt_attn_proj',
            r'mlp_txt\.gate_and_up_proj:txt_mlp.linear_fc1',
            r'mlp_txt\.down_proj:txt_mlp.linear_fc2',
            r'mod_proj:img_mod',
            r'self_attn\.query_layernorm:img_attn_q_norm',
            r'self_attn\.key_layernorm:img_attn_k_norm',
            r'self_attn\.q_proj:img_attn_q',
            r'self_attn\.k_proj:img_attn_k',
            r'self_attn\.v_proj:img_attn_v',
            r'self_attn\.o_proj:img_attn_proj',
            r'mlp\.gate_and_up_proj:img_mlp.linear_fc1',
            r'mlp\.down_proj:img_mlp.linear_fc2',
            r'patch_embed:img_in',
            r'text_projector:txt_in',
            r'time_embed:time_in',
        ])
    ],
    "leo2_moe_ptm2_to_fsdp2": [
        # Map the fsdp model keys to the ptmv2 ckpt keys.
        '|'.join([  # List order matters.
            r'layers:layers.layers',
            r'mlp_txt\.gate_and_up_proj:mlp_txt.linear_fc1',
            r'mlp_txt\.down_proj:mlp_txt.linear_fc2',
            r'mlp_audio\.gate_and_up_proj:mlp_audio.linear_fc1',
            r'mlp_audio\.down_proj:mlp_audio.linear_fc2',
            r'gate\.wg:router',
            r'expert_gate_and_up_weights:experts.experts.linear_fc1.weight',
            r'expert_down_weights:experts.experts.linear_fc2.weight',
            r'shared_mlp\.gate_and_up_proj:shared_experts.linear_fc1',
            r'shared_mlp\.down_proj:shared_experts.linear_fc2',
            r'mlp\.gate_and_up_proj:mlp.linear_fc1',
            r'mlp\.down_proj:mlp.linear_fc2',
        ])
    ],
    "leo2_moe_ptm2_to_fsdp2_epmoe": [
        '|'.join([  # List order matters.
            r'layers:layers.layers',
            r'mlp_txt\.gate_and_up_proj:mlp_txt.linear_fc1',
            r'mlp_txt\.down_proj:mlp_txt.linear_fc2',
            r'mlp_audio\.gate_and_up_proj:mlp_audio.linear_fc1',
            r'mlp_audio\.down_proj:mlp_audio.linear_fc2',
            r'gate\.wg:router',
            r'experts\.expert_gate_and_up_weights:experts.experts.linear_fc1.weight',
            r'experts\.expert_down_weights:experts.experts.linear_fc2.weight',
            r'shared_mlp\.gate_and_up_proj:shared_experts.linear_fc1',
            r'shared_mlp\.down_proj:shared_experts.linear_fc2',
            r'mlp\.gate_and_up_proj:mlp.linear_fc1',
            r'mlp\.down_proj:mlp.linear_fc2',
        ])
    ]
}
