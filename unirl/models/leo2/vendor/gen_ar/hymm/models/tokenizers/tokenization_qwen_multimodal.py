from transformers import AutoTokenizer, AddedToken
from .tokenization_hunyuan_multimodal import HunyuanMultimodalTokenizerFast


class QwenMultimodalTokenizerFast(HunyuanMultimodalTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_special_tokens(self):
        # Define names for commonly used special tokens
        predefined_name_mapping = {
            "bos": "<|im_start|>",
            "eos": "<|im_end|>",
            "think": "<think>",
            "end_of_think": "</think>",
            "answer": "",
            "end_of_answer": "",
            "boi": "<|vision_start|>",
            "eoi": "<|vision_end|>",
            "img": "<|image_pad|>",
            "bov": "<|vision_start|>",
            "eov": "<|vision_end|>",
            "video": "<|video_pad|>",
            "quad": "<|quad_start|>",
            "end_of_quad": "<|quad_end|>",
            "ref": "<|object_ref_start|>",
            "end_of_ref": "<|object_ref_end|>",
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


class Qwen2TokenizerFast(HunyuanMultimodalTokenizerFast):
    # NOTICE: Don't create __init__ here, because from transformers>=5, it will determine if the class is a legacy
    # tokenizer by checking if __init__ is not defined. Here we should use a legacy tokenizer.

    def setup_special_tokens(self):
        name_mapping = dict(
            img_token="<|image_pad|>",
            video_token="<|video_pad|>",
            audio_token="<|audio_pad|>",
            cfg_token="<|im_end|>",
        )

        missing_tokens = [t for t in name_mapping.values() if t not in self._sp_dict]
        if missing_tokens:
            self.add_tokens(
                [AddedToken(t, special=True, normalized=False) for t in missing_tokens],
                special_tokens=True,
            )
            for t in missing_tokens:
                self._sp_dict[t] = self.convert_tokens_to_ids(t)

        for name, token in name_mapping.items():
            setattr(self, name, token)
            setattr(self, f"{name}_id", self._sp_dict[token])

        # Qwen-VL family (Qwen2-VL / Qwen2.5-VL / Qwen3-VL) ships <|vision_start|> / <|vision_end|>
        # for wrapping image/video tokens. They are not present in pure-text Qwen2/Qwen3,
        # so only register them when the underlying vocabulary actually has them.
        vl_name_mapping = dict(
            boi_token="<|vision_start|>",
            eoi_token="<|vision_end|>",
            bov_token="<|vision_start|>",
            eov_token="<|vision_end|>",
        )
        for name, token in vl_name_mapping.items():
            tok_id = self._sp_dict.get(token)
            if tok_id is None:
                continue
            setattr(self, name, token)
            setattr(self, f"{name}_id", tok_id)

class Qwen3TokenizerFast(Qwen2TokenizerFast):

    def setup_special_tokens(self):
        name_mapping = dict(
            img_token="<|image_pad|>",
            video_token="<|video_pad|>",
            audio_token="<|audio_pad|>",
            cfg_token="<|cfg|>",
            timestep_token="<|timestep|>",
            timestep_r_token="<|timestep_r|>",
            guidance_token="<|guidance|>",
        )

        missing_tokens = [t for t in name_mapping.values() if t not in self._sp_dict]
        if missing_tokens:
            self.add_tokens(
                [AddedToken(t, special=True, normalized=False) for t in missing_tokens],
                special_tokens=True,
            )
            for t in missing_tokens:
                self._sp_dict[t] = self.convert_tokens_to_ids(t)

        for name, token in name_mapping.items():
            setattr(self, name, token)
            setattr(self, f"{name}_id", self._sp_dict[token])

        # Qwen3.5 (and Qwen3-VL family) uses <|vision_start|>/<|vision_end|> to wrap
        # image/video patch tokens when multimodal input is present.
        vl_name_mapping = dict(
            boi_token="<|vision_start|>",
            eoi_token="<|vision_end|>",
            bov_token="<|vision_start|>",
            eov_token="<|vision_end|>",
        )
        for name, token in vl_name_mapping.items():
            tok_id = self._sp_dict.get(token) or self.convert_tokens_to_ids(token)
            unk_id = getattr(self, "unk_token_id", None)
            if tok_id is None or tok_id == unk_id:
                continue
            setattr(self, name, token)
            setattr(self, f"{name}_id", tok_id)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        try:
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except ValueError as err:
            msg = str(err)
            if "Couldn't instantiate the backend tokenizer" not in msg:
                raise
            # used by AutoTokenizer to construct the tokenizer, 
            # the directory may not have tokenizer.json (Qwen3-Omni etc.)
            hf_tok = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                trust_remote_code=True,
                **{k: v for k, v in kwargs.items() if k not in ("trust_remote_code",)},
            )
            if not hasattr(hf_tok, "_tokenizer"):
                raise RuntimeError(
                    f"AutoTokenizer 从 {pretrained_model_name_or_path} 加载的 tokenizer "
                    f"({type(hf_tok).__name__}) 没有 _tokenizer 属性（非 fast tokenizer）。"
                ) from err
            init_kw = dict(getattr(hf_tok, "init_kwargs", None) or {})
            for k in ("tokenizer_file", "tokenizer_object", "vocab_file"):
                init_kw.pop(k, None)
            for attr in ("bos_token", "eos_token", "unk_token", "pad_token"):
                val = getattr(hf_tok, attr, None)
                if val is not None:
                    init_kw.setdefault(attr, val)
            if hasattr(hf_tok, "added_tokens_decoder") and hf_tok.added_tokens_decoder:
                init_kw.setdefault("added_tokens_decoder", hf_tok.added_tokens_decoder)
            return cls(tokenizer_object=hf_tok._tokenizer, **init_kw)
