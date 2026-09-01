from .tokenization_hunyuan_multimodal import HunyuanMultimodalTokenizerFast


class Hunyuan3MultimodalV3d5TokenizerFast(HunyuanMultimodalTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_special_tokens(self):
        # Define names for commonly used special tokens
        predefined_name_mapping = {
            "think": "<think>",  # 思考内容  120167
            "end_of_think": "</think>",  # 思考内容结束  120168
            "answer": "",  # 答案内容
            "end_of_answer": "",  # 答案内容结束
            "boi": "<｜hy_image▁start｜>",  # 图片开始  120118
            "eoi": "<｜hy_image▁end｜>",  # 图片结束  120119
            "img": "<｜hy_image▁pad｜>",  # 120120
            "bov": "<｜hy_video▁start｜>",  # 120122
            "eov": "<｜hy_video▁end｜>",   # 120123
            "video": "<｜hy_video▁pad｜>",  # 120683
            "box": "<box>",  # 120126
            "end_of_box": "</box>",  # 120127
            "quad": "<quad>",  # 120128
            "end_of_quad": "</quad>",  # 120129
            "ref": "<ref>",  # 120130
            "end_of_ref": "</ref>",  # 120131
            "pFig": "<pFig>",  # 120132
            "end_of_pFig": "</pFig>",  # 120133
            "det": "<det>",  # 120134
            "end_of_det": "</det>",  # 120135
            "point": "<point>",  # 120136
            "end_of_point": "</point>",  # 120137
        }
        for name, mapping in predefined_name_mapping.items():
            setattr(self, f"{name}_token", mapping)
            setattr(self, f"{name}_token_id", self.convert_tokens_to_ids(mapping))

        if len(self._sp_dict) > 0:
            name_mapping = dict(
                boa_token="<｜boa｜>",
                eoa_token="<｜eoa｜>",
                audio_token="<｜audio｜>",
                cfg_token="<｜cfg｜>",
                timestep_token="<｜timestep｜>",
                guidance_token="<｜guidance｜>",
                joint_img_sep_token="<｜joint_img_sep｜>",
                # for extended cot types
                recaption_token="<｜recaption｜>",
                end_of_recaption_token="<｜end_of_recaption｜>",
            )
            for name, token in name_mapping.items():
                setattr(self, name, token)
                setattr(self, f"{name}_id", self._sp_dict[token])
