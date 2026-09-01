import argparse
from pathlib import Path
from loguru import logger

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def extend_tokenizer_with_additional_special_tokens(
        tokenizer_path: str,
        save_path: str,
        special_tokens: dict,
):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    existing_additional_special_tokens = tokenizer.special_tokens_map.get('additional_special_tokens', [])
    # replace_additional_special_tokens defaults to True in transformers, and it will replace the existing additional special tokens with new ones.
    # set it to False to append new tokens to existing ones.
    if len(existing_additional_special_tokens) > 0:
        logger.warning(f"Additional special tokens already exist in the tokenizer. Adding new tokens without replacing existing ones.")
        tokenizer.add_special_tokens(special_tokens, replace_additional_special_tokens=False)
    else:
        tokenizer.add_special_tokens(special_tokens)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(save_path)
    print(f"Saved tokenizer at {save_path}")


VERSIONS = dict(
    # Extend ratio tokens
    v3=dict(
        sizes=[32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192],
        ratio_buffer_size=256,
    ),
    # Support video
    v4=dict(
        sizes=[32, 64, 128, 256, 384, 512, 640, 768, 960, 1024, 1280, 1536, 1920, 2048, 2560, 3072, 3584, 4096, 5120,
               6144, 7168, 8192],
        ratio_buffer_size=256,
        duration_buffer_size=256,
    ),
    v5=dict(
        sizes=[32, 64, 128, 256, 384, 512, 640, 768, 960, 1024, 1280, 1536, 1920, 2048, 2560, 3072, 3584, 4096, 5120,
               6144, 7168, 8192],
        ratio_buffer_size=256,
        duration_buffer_size=256,
        # for add_tw_th_token: tw_th = token length per dimension (h_token, w_token)
        tw_th=list(range(1,513)), # [16, 8192]
    ),
)
VERSIONS["hymm_v3_5"] = VERSIONS["v4"]
VERSIONS["hymm_v3_5_v2"] = VERSIONS["v5"]

def extend_hunyuan_multimodal(tokenizer_path: str, save_path: str, version: str):
    special_tokens = [
        # for multimedia inputs
        "<｜boi｜>",  # begin of image
        "<｜eoi｜>",  # end of image
        "<｜boa｜>",  # begin of audio
        "<｜eoa｜>",  # end of audio
        "<｜bov｜>",  # begin of video
        "<｜eov｜>",  # end of video
        "<｜img｜>",
        "<｜audio｜>",
        "<｜video｜>",
        "<｜cfg｜>",  # classifier free guidance
        "<｜timestep｜>",
        "<｜timestep_r｜>",   # meanflow
        "<｜guidance｜>",
        "<｜joint_img_sep｜>",
        # for extended cot types
        "<｜recaption｜>",
        "<｜end_of_recaption｜>",
        # for grounding
        "<｜ref｜>",
        "<｜end_of_ref｜>",
        "<｜quad｜>",
        "<｜end_of_quad｜>",
    ]

    # for image sizes
    image_size_buffer_size = 32
    sizes = VERSIONS[version]["sizes"]
    unused_id = 0
    for i in range(image_size_buffer_size):
        if i < len(sizes):
            size = sizes[i]
        else:
            size = f"unused_{unused_id}"
            unused_id += 1
        special_tokens.append(f"<｜img_size_{size}｜>")

    # for image ratios
    image_ratio_buffer_size = VERSIONS[version]["ratio_buffer_size"]
    for i in range(image_ratio_buffer_size):
        special_tokens.append(f"<｜img_ratio_{i}｜>")

    # for image tw_th (v35: h_token, w_token as token length per dimension)
    if "tw_th" in VERSIONS.get(version, {}):
        tw_th = VERSIONS[version]["tw_th"]
        tw_th_buffer_size = VERSIONS[version].get("tw_th_buffer_size", len(tw_th))
        unused_id = 0
        for i in range(tw_th_buffer_size):
            k = tw_th[i] if i < len(tw_th) else f"unused_{unused_id}"
            if i >= len(tw_th):
                unused_id += 1
            special_tokens.append(f"<｜img_tw_th_{k}｜>")

    if "duration_buffer_size" in VERSIONS[version]:
        # for video durations
        duration_buffer_size = VERSIONS[version]["duration_buffer_size"]
        for i in range(duration_buffer_size):
            special_tokens.append(f"<｜duration_{i}｜>")

    # for relation pairs
    relation_pair_buffer_size = 32
    for i in range(relation_pair_buffer_size):
        special_tokens.extend([f"<｜relation_{i}｜>", f"<｜end_of_relation_{i}｜>"])

    # for positions
    xyz_buffer_sizes = dict(x=2049, y=2049, z=1025)
    for axis, buffer_size in xyz_buffer_sizes.items():
        for i in range(buffer_size):
            special_tokens.append(f"<｜pos_{axis}_{i}｜>")

    # Finally, extend the tokenizer
    extend_tokenizer_with_additional_special_tokens(
        tokenizer_path=tokenizer_path,
        save_path=save_path,
        special_tokens=dict(
            additional_special_tokens=special_tokens,
        ),
    )


def extend_hymm_v3_5(tokenizer_path: str, save_path: str, version: str):
    # Extend from HY3.0 tokenizer.
    # Add special tokens for multimodal inputs.

    special_tokens = [
        # for multimedia inputs
        "<｜boa｜>",  # begin of audio
        "<｜eoa｜>",  # end of audio
        "<｜bov｜>",  # begin of video
        "<｜eov｜>",  # end of video
        "<｜audio｜>",
        "<｜video｜>",
        "<｜cfg｜>",  # classifier free guidance
        "<｜timestep｜>",
        "<｜timestep_r｜>",   # meanflow
        "<｜guidance｜>",
        "<｜joint_img_sep｜>",
        # for recaption. Will not used anymore. Just for compatibility.
        "<｜recaption｜>",
        "<｜end_of_recaption｜>",
        # for grounding
        "<｜ref｜>",
        "<｜end_of_ref｜>",
        "<｜quad｜>",
        "<｜end_of_quad｜>",
    ]

    # for image sizes
    image_size_buffer_size = 32
    sizes = VERSIONS[version]["sizes"]
    unused_id = 0
    for i in range(image_size_buffer_size):
        if i < len(sizes):
            size = sizes[i]
        else:
            size = f"unused_{unused_id}"
            unused_id += 1
        special_tokens.append(f"<｜img_size_{size}｜>")

    # for image ratios
    image_ratio_buffer_size = VERSIONS[version]["ratio_buffer_size"]
    for i in range(image_ratio_buffer_size):
        special_tokens.append(f"<｜img_ratio_{i}｜>")

    # for image tw_th (v35: h_token, w_token as token length per dimension)
    if "tw_th" in VERSIONS.get(version, {}):
        tw_th = VERSIONS[version]["tw_th"]
        tw_th_buffer_size = VERSIONS[version].get("tw_th_buffer_size", len(tw_th))
        unused_id = 0
        for i in range(tw_th_buffer_size):
            k = tw_th[i] if i < len(tw_th) else f"unused_{unused_id}"
            if i >= len(tw_th):
                unused_id += 1
            special_tokens.append(f"<｜img_tw_th_{k}｜>")

    if "duration_buffer_size" in VERSIONS[version]:
        # for video durations
        duration_buffer_size = VERSIONS[version]["duration_buffer_size"]
        for i in range(duration_buffer_size):
            special_tokens.append(f"<｜duration_{i}｜>")

    # for relation pairs
    relation_pair_buffer_size = 32
    for i in range(relation_pair_buffer_size):
        special_tokens.extend([f"<｜relation_{i}｜>", f"<｜end_of_relation_{i}｜>"])

    # for positions
    xy_buffer_sizes = dict(x=1001, y=1001)
    for axis, buffer_size in xy_buffer_sizes.items():
        for i in range(buffer_size):
            special_tokens.append(f"<｜{axis}_{i}｜>")

    # Finally, extend the tokenizer
    extend_tokenizer_with_additional_special_tokens(
        tokenizer_path=tokenizer_path,
        save_path=save_path,
        special_tokens=dict(
            additional_special_tokens=special_tokens,
        ),
    )
    # Finally, manually modify following tokens:
    # <｜hy_place▁holder▁no▁100｜>  -->  <｜boi｜>
    # <｜hy_place▁holder▁no▁101｜>  -->  <｜eoi｜>
    # <｜hy_place▁holder▁no▁102｜>  -->  <｜img｜>


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to the original tokenizer.")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the extended tokenizer.")
    parser.add_argument("--version", type=str, required=True, help="Version of the tokenizer.")
    args = parser.parse_args()

    if "hymm_v3_5" in args.version:
        extend_hymm_v3_5(
            tokenizer_path=args.tokenizer_path,
            save_path=args.save_path,
            version=args.version,
        )
    else:
        extend_hunyuan_multimodal(
            tokenizer_path=args.tokenizer_path,
            save_path=args.save_path,
            version=args.version,
        )


if __name__ == "__main__":
    main()


# python3 hymm/models/tokenizers/utils.py \
#      --tokenizer-path /apdcephfs_zwfy/share_303937731/1_public_models/hymm_ar_assets/pretrained_llm/HunyuanMoE-A3B-MidTrain \
#      --save-path /apdcephfs_zwfy/share_303937731/1_public_models/hymm_ar_assets/pretrained_llm/extended_tokenizer/hunyuan_multimodal_v4/ \
#      --version v4

# python3 hymm/models/tokenizers/utils.py \
#      --tokenizer-path /apdcephfs_zwfy/share_303937731/jarvizhang/workspace/HunYuanTokenizer/tokenizer \
#      --save-path /apdcephfs_zwfy/share_303937731/1_public_models/hymm_ar_assets/pretrained_llm/extended_tokenizer/hymm_v3_5/ \
#      --version hymm_v3_5

# python3 hymm/models/tokenizers/utils.py \
#      --tokenizer-path /apdcephfs_wza/jarvizhang/workspace/HunYuanTokenizer/tokenizer \
#      --save-path /apdcephfs_wza/1_public_models/hymm_ar_assets/pretrained_llm/extended_tokenizer/hymm_v3_5_v2/ \
#      --version hymm_v3_5_v2

# python3 hymm/models/tokenizers/utils.py \
#      --tokenizer-path /apdcephfs_wza/1_public_models/hymm_ar_assets/pretrained_llm/v3_a3b_offical_ckpts/256k/hf \
#      --save-path /apdcephfs_wza/1_public_models/hymm_ar_assets/pretrained_llm/extended_tokenizer/hunyuan3_moe_a3b_hymm_v3_5 \
#      --version hymm_v3_5_v2
