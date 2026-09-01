import os
from ctypes import resize
from PIL import Image
from io import BytesIO
from typing import Union, List

from transformers import AltCLIPModel, AltCLIPProcessor
from torch import nn
import torch



class AltCLIPRM(nn.Module):
    def __init__(self, ft_model_path, pretrained_model_name_or_path="BAAI/AltCLIP", resize_res=224, processor_cache_dir=None, http_proxy=None, https_proxy=None):
        super().__init__()

        self.processor = AltCLIPProcessor.from_pretrained(pretrained_model_name_or_path, cache_dir=processor_cache_dir, use_fast=False)
        print(f"Loading precessor {pretrained_model_name_or_path} done.")
        self.model = AltCLIPModel.from_pretrained(ft_model_path, dtype=torch.bfloat16)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        # set image size
        self.image_processor.crop_size = {"height": resize_res, "width": resize_res}
        self.image_processor.size["shortest_edge"] = resize_res

        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy


    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)


    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)


    def forward(self, text_inputs=None, image_inputs=None):
        outputs = ()
        if text_inputs is not None:
            outputs += self.model.get_text_features(**text_inputs),
        if image_inputs is not None:
            outputs += self.model.get_image_features(image_inputs, interpolate_pos_encoding=True),  # compatible with different input shapes
        return outputs


    @property
    def logit_scale(self):
        return self.model.logit_scale

    def load(self, path):
        self.model = AltCLIPModel.from_pretrained(path, dtype=torch.bfloat16, weights_only=False)

    def save(self, path):
        self.model.save_pretrained(path)


    def tokenize(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            # max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return text_inputs


    def process_image(self, images):
        if not isinstance(images, list):
            images = [images]

        image_inputs = []
        for image in images:
            if isinstance(image, dict):
                image = image["bytes"]
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, str):
                image = Image.open(image)
            assert isinstance(image, Image.Image), "image must be PIL.Image"
            image = image.convert("RGB")
            image_inputs.append(image) 
        pixel_values = self.image_processor(image_inputs, return_tensors="pt")["pixel_values"]
        return pixel_values


    def get_preference_scores(self, prompt, images):
        image_inputs = self.process_image(images).to(self.model.device)
        text_inputs = self.tokenize(prompt).to(self.model.device)

        
        with torch.inference_mode():
            text_embs, image_embs = self.forward(text_inputs, image_inputs)
            
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            rm_scores = self.logit_scale.exp() * (text_embs @ image_embs.T).flatten()
            
            probs = torch.softmax(rm_scores, dim=-1)
        
        return rm_scores, probs
    

    def __call__(
            self,
            images: Union[Image.Image, List[Image.Image]],
            texts: Union[str, List[str]],
    ):
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]
        
        rm_scores, probs = self.get_preference_scores(texts, images)
        
        rm_scores = rm_scores.cpu().tolist()
        probs = probs.cpu().tolist()
        return rm_scores


if __name__ == "__main__":
    device = "cuda"
    # model_path_224 = "/apdcephfs_nj10/share_301739632/taoxxzhang/exp/aes_rm/hunyuan_aes_rm/rm_005_altclip_224_aes1.1w_deformity11w_lr3e-6_step4000_bs16_gpu4_bt_clip0.1_pos/checkpoint-final"
    # altclip_224 = AltCLIPRM(model_path_224, resize_res=224)
    # altclip_224.eval()
    # altclip_224.to(device)

    # model_path_512 = "/apdcephfs_nj10/share_301739632/taoxxzhang/exp/aes_rm/hunyuan_aes_rm/rm_006_altclip_512_aes1.1w_deformity11w_lr3e-6_step4000_bs4_gradacc2_gpu8_bt_clip0.1_pos/checkpoint-final"
    # altclip_512_006 = AltCLIPRM(model_path_512, resize_res=512)
    # altclip_512_006.eval()
    # altclip_512_006.to(device)

    model_path_512 = "/apdcephfs_zwfy2/share_303937731/yutaocui/model_zoo/bt_clip_zt"
    processor_cache_dir = "/apdcephfs_zwfy2/share_303937731/yutaocui/model_zoo/hf_cache/hub"
    # model_path_512 = "/apdcephfs_nj10/share_301739632/taoxxzhang/exp/aes_rm/hunyuan_aes_rm/rm_007_altclip_512_aes1.1w_deformity11w_aes15w0828_deformity500yt0902_lr3e-6_step4000_bs4_gradacc2_gpu8_bt_clip0.1_pos/checkpoint-final"
    altclip_512_007 = AltCLIPRM(
        model_path_512,
        pretrained_model_name_or_path="/apdcephfs_zwfy2/share_303937731/yutaocui/model_zoo/AltCLIP",
        resize_res=512,
        processor_cache_dir=processor_cache_dir,
        # http_proxy="http://9.21.0.122:11113",
        # https_proxy="http://9.21.0.122:11113"
    )
    altclip_512_007.eval()
    altclip_512_007.to(device)

    prompts_list = [
        ["Medium shot, Realistic Photography Style, The lighting is soft and even, Neutral, The background is a wooden table. A hand with a patterned watchband rests on a wooden table."],
        ["A young woman with long brown hair styled in a high bun stands against a plain white background. She wears a white t-shirt and olive green cargo overalls. The overalls feature multiple pockets and adjustable straps. Her hands are casually placed in her pockets. She wears white sneakers with a logo on the side. The lighting is bright and even, creating a clean and modern aesthetic. The lighting is bright and even, Centered composition, Neutral, Full shot, The background is plain white, Realistic Photography Style, True-to-life, stunning, detail-rich, detail-rich, Ultra HD"],
        ["The lighting is bright and even, Photorealistic Photography Style, Peaceful, The background is a light wood floor, High angle close-up shot. A black and white dog lies near a red paper fan with a gold circular design., Photorealistic"],
        ["""A person is shown from the waist down, wearing a pair of swim trunks and sandals. The swim trunks feature horizontal stripes in orange, white, light blue, and navy blue. The waistband is a bright orange, and the drawstring is also orange. The person's right hand is in their pocket. The sandals are navy blue with a white strap across the top. The strap has "NAUTICA" written in white. The person's left hand is relaxed at their side. The background is plain white. Neutral, Realistic Photography Style, Medium shot, The background is plain white, The lighting is bright and even., Naturalistic, 4k"""],
        ["A close-up selfie features a man with short, graying hair and a goatee. He wears a camouflage tank top and a blue patterned neck gaiter. The man's face is centered in the frame, and he appears to be smiling slightly. The background includes a paved area, a building with a tan facade, and a line of people standing near the building entrance. A dark-colored car is partially visible in the lower right corner of the image. The lighting is bright, suggesting a sunny day. A small tattoo is visible on the man's left shoulder. A small sign is visible in the background near the car. Close-up shot, The background includes a paved area, a building, and other people, The lighting is bright and natural, likely sunlight, Casual, Realistic Photography Style, Naturalistic, high resolution, Photorealistic, high clarity, True-to-life"]
    ]
    images_list = [
        ["/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/18_0.png",
         "/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/18_1.png"],
        ["/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/89_0.png",
         "/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/89_1.png"],
        ["/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/206_0.png",
         "/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/206_1.png"],
        ["/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/186_0.png",
         "/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/186_1.png"],
        ["/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/4_0.png",
         "/apdcephfs_nj10/share_301739632/taoxxzhang/code/aes_rm/trainer/eval_imgs/4_1.png"]
    ]

    with torch.no_grad():
        for prompts, images in zip(prompts_list, images_list):
            # rm_scores_224, probs_224 = altclip_224.get_preference_scores(prompts, images)
            # rm_scores_512_006, probs_512_006 = altclip_512_006.get_preference_scores(prompts, images)
            # rm_scores_512_007, probs_512_007 = altclip_512_007.get_preference_scores(prompts, images)
            rm_scores_512_007, probs_512_007 = altclip_512_007(images, prompts)
            # print(prompts[0])
            # print(rm_scores_224, probs_224)
            # print(rm_scores_512_006, probs_512_006)
            print(rm_scores_512_007, probs_512_007)
            print("\n")

"""
pssh -i -t 0 -h /root/hosts pip install transformers==4.56.1
pssh -i -t 0 -h /root/hosts python3 /apdcephfs_nj10/share_301739632/yutaocui/workspace/hunyuan_multimoda_gen_ar/hymm/models/reward_models/altclip_rm.py
"""
