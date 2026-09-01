from .tokenization_hunyuan_multimodal import HunyuanMultimodalTokenizerFast


class HunyuanMultimodalV3d5TokenizerFast(HunyuanMultimodalTokenizerFast):

    def setup_special_tokens(self):
        # Define names for commonly used special tokens
        predefined_name_mapping = {
            "think": "<think>",
            "end_of_think": "</think>",
            "answer": "<answer>",           # not used
            "end_of_answer": "</answer>",   # not used
            "boi": "<｜boi｜>",
            "eoi": "<｜eoi｜>",
            "img": "<｜img｜>",
        }
        for name, mapping in predefined_name_mapping.items():
            setattr(self, f"{name}_token", mapping)
            setattr(self, f"{name}_token_id", self.convert_tokens_to_ids(mapping))

        if len(self._sp_dict) > 0:
            name_mapping = dict(
                boa_token="<｜boa｜>",
                eoa_token="<｜eoa｜>",
                bov_token="<｜bov｜>",
                eov_token="<｜eov｜>",
                audio_token="<｜audio｜>",
                video_token="<｜video｜>",
                cfg_token="<｜cfg｜>",
                timestep_token="<｜timestep｜>",
                timestep_r_token="<｜timestep_r｜>",
                guidance_token="<｜guidance｜>",
                joint_img_sep_token="<｜joint_img_sep｜>",
                # for extended cot types
                recaption_token="<｜recaption｜>",
                end_of_recaption_token="<｜end_of_recaption｜>",
                # for grounding
                ref_token="<｜ref｜>",
                end_of_ref_token="<｜end_of_ref｜>",
                quad_token="<｜quad｜>",
                end_of_quad_token="<｜end_of_quad｜>",
            )
            for name, token in name_mapping.items():
                setattr(self, name, token)
                setattr(self, f"{name}_id", self._sp_dict[token])

