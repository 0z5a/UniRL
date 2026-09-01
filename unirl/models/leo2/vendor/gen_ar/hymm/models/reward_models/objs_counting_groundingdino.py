import re
from typing import List
from types import SimpleNamespace

import PIL
import torch
from PIL import Image

try:
    from word2number import w2n
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util.utils import clean_state_dict
except:
    Warning("'groundingdino' and 'word2number' not installed, please install them for using the rewrad models.")
    pass


GROUNDINGDINO_ARGS = SimpleNamespace(
    batch_size=1,
    modelname="groundingdino",
    backbone="swin_T_224_1k",
    position_embedding="sine",
    pe_temperatureH=20,
    pe_temperatureW=20,
    return_interm_indices=[1, 2, 3],
    backbone_freeze_keywords=None,
    enc_layers=6,
    dec_layers=6,
    pre_norm=False,
    dim_feedforward=2048,
    hidden_dim=256,
    dropout=0.0,
    nheads=8,
    num_queries=900,
    query_dim=4,
    num_patterns=0,
    num_feature_levels=4,
    enc_n_points=4,
    dec_n_points=4,
    two_stage_type="standard",
    two_stage_bbox_embed_share=False,
    two_stage_class_embed_share=False,
    transformer_activation="relu",
    dec_pred_bbox_embed_share=True,
    dn_box_noise_scale=1.0,
    dn_label_noise_ratio=0.5,
    dn_label_coef=1.0,
    dn_bbox_coef=1.0,
    embed_init_tgt=True,
    dn_labelbook_size=2000,
    max_text_len=256,
    text_encoder_type="bert-base-uncased",
    use_text_enhancer=True,
    use_fusion_layer=True,
    use_checkpoint=True,
    use_transformer_ckpt=True,
    use_text_cross_attention=True,
    text_dropout=0.0,
    fusion_dropout=0.0,
    fusion_droppath=0.1,
    sub_sentence_present=True,
)
MODEL_CKP_PATH = "/apdcephfs_gy2/share_302507476/yutaocui/model_zoo/groundingdino_weights/groundingdino_swint_ogc.pth"
NUMS = [
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]


def parse_caption_numbers(caption):
    try:
        matches = re.findall(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
            r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
            r")(?:-?\w+)*\b",
            caption.lower(),
        )

        numbers = []
        for word in matches:
            try:
                numbers.append(w2n.word_to_num(word))
            except:
                if word.isdigit():
                    numbers.append(int(word))

        unique_numbers = list(set(numbers))
        return unique_numbers[0] if len(unique_numbers) == 1 else None
    except Exception as e:
        print(f"Error parsing caption: {e}")
        return None


class ObjectsCountingGroundingDino(object):
    def __init__(self, box_thre=0.5, device="cuda"):
        self.model = self.load_groudingdino_model(MODEL_CKP_PATH, device)
        self.box_thre = box_thre
        self.device = device

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def load_groudingdino_model(model_checkpoint_path, device):
        args = GROUNDINGDINO_ARGS
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

        return model.to(device).eval()

    def load_image_and_transform(self, image_pil=None, img_path=None):
        if image_pil is None:
            assert img_path is not None
            image_pil = Image.open(img_path).convert("RGB")  # load image

        img_tensor, _ = self.transform(image_pil, None)  # 3, h, w
        return img_tensor

    def pred_objs_num(self, images_pil: List[PIL.Image.Image], objs: List[str]):
        objs_caption = [obj + "." for obj in objs]  # add '.' to avoid groundingdino empty special token masks error
        imgs_tensor = [self.load_image_and_transform(img_pil) for img_pil in images_pil]
        imgs_tensor = torch.stack(imgs_tensor).to(self.device)  # B, C, H, W
        with torch.no_grad():
            outputs = self.model(imgs_tensor, captions=objs_caption)

        pred_nums = []
        for i in range(imgs_tensor.shape[0]):
            logits = outputs["pred_logits"].sigmoid()[i]  # (nq, 256)
            boxes = outputs["pred_boxes"][i]  # (nq, 4)

            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > self.box_thre
            eff_num = filt_mask.int().sum()
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            pred_nums.append(eff_num.item())

        return pred_nums

    @staticmethod
    def check_format(answer):
        if not isinstance(answer, str):
            return False

        pattern = r"""
            ^<answer>
            <gen_boi>
            (<gen_img>)+
            <gen_eoi>
            </answer>$
        """

        if re.fullmatch(pattern, answer, re.VERBOSE):
            return True
        else:
            return False

    def __call__(
        self,
        images_pil: List[PIL.Image.Image],
        objs: List[str],
        prompts: List[str],
        input_prompts: List[str],
        extra_und_penalty: bool = False,
        enable_think_mode: bool = False,
    ):
        pred_nums = self.pred_objs_num(images_pil, objs)
        rewards = []
        assert len(prompts) == len(input_prompts)

        for pred_num, prompt, input_prompt in zip(pred_nums, prompts, input_prompts):
            prompt_num = parse_caption_numbers(prompt)
            input_prompt_num = parse_caption_numbers(input_prompt)
            if not enable_think_mode:
                # 只有当pred_num == prompt中的唯一数字且 == input_prompt中的唯一数字时，reward=1, 否则reward=0
                if prompt_num is None:
                    rewards.append(0)
                    continue

                if prompt_num == pred_num and pred_num == input_prompt_num:
                    rewards.append(1)
                elif extra_und_penalty and pred_num != prompt_num:
                    rewards.append(-1)
                else:
                    rewards.append(0)
            else:
                if self.check_format(prompt) and pred_num == input_prompt_num:
                    rewards.append(1)
                else:
                    rewards.append(0)

        return rewards


if __name__ == "__main__":
    # groundingdino_reward_func = ObjectsCountingGroundingDino(box_thre=0.5)

    last_answer = "<answer><gen_boi><gen_img><gen_img><gen_eoi></answer>"
    result = ObjectsCountingGroundingDino().check_format(last_answer)
    print(result)
