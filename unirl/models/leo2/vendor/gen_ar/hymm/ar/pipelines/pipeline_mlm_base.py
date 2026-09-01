from typing import List, Tuple, Optional

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ..mask_schedulers import create_attention_mask_general
from ...utils.torch_utils import PRECISION_TO_TYPE


class MaskedImageModelingPipeline(DiffusionPipeline):

    def guidance_scale(self, step):
        if self._guidance_dynamic == 'const':
            guidance = self._guidance_scale
        elif self._guidance_dynamic == 'linear_inc':
            guidance = (self._guidance_scale - 1) * (step / (self.num_timesteps - 1)) + 1
        elif self._guidance_dynamic == 'linear_dec':
            guidance = (self._guidance_scale - 1) * ((self.num_timesteps - 1 - step) / (self.num_timesteps - 1)) + 1
        else:
            raise ValueError(f"Unknown guidance dynamic: {self._guidance_dynamic}")
        return guidance

    def prepare_tokens_and_mask(self, batch_size, height, width, device):
        shape = (batch_size, height, width)
        init_tokens = torch.full(shape, self.arrange.mask_id).to(device)
        init_mask = torch.ones(shape).to(device)

        return init_tokens, init_mask

    def prepare_instruction_tuning_whole_tokens(
            self,
            prompts: List[Tuple[str]],
            src_shifted_image_tokens: torch.LongTensor,
            tgt_masked_image_tokens: torch.LongTensor,
            image_token_len: int,
            negative_prompts: Optional[List[str]] = None,
            do_classifier_free_guidance: bool = False,
            device: Optional[torch.device] = None,
            **kwargs,
    ):
        """
        Each prompts element is a tuple of strings:
        - (prompt, ) for editing task
        - (instruction, prompt) for inpainting task

        Returns
        -------
        batch_whole_tokens: torch.LongTensor
            [batch_size, text_token_len + image_token_len], the whole tokens to predict
        batch_uncond_whole_tokens: torch.LongTensor
            [batch_size, text_token_len + image_token_len], the whole tokens to predict unconditionally
        """
        batch_whole_tokens = []
        for prompt, src_shifted_image_token, tgt_masked_image_token in zip(prompts, src_shifted_image_tokens, tgt_masked_image_tokens):
            whole_tokens = self.tokenizer.encode_mlm_editing(
                *prompt,
                src_masked_shifted_image_tokens=src_shifted_image_token,
                tgt_masked_shifted_image_tokens=tgt_masked_image_token,
                src_image_token_len=image_token_len,
                tgt_image_token_len=image_token_len,
                max_text_token_length=self.arrange.s_text_maxlen,
                max_total_token_length=self.arrange.s_text_maxlen + 2 * image_token_len,
                instruction_uncond_p=0,
                return_labels=False,
                **kwargs,
            )[:-1]  # :-1 remove the eos token
            batch_whole_tokens.append(whole_tokens)
        batch_whole_tokens = torch.stack(batch_whole_tokens, dim=0)
        batch_whole_tokens = batch_whole_tokens.to(device)

        if do_classifier_free_guidance:
            batch_uncond_whole_tokens = []
            for prompt, neg_prompt, src_shifted_image_token, tgt_masked_image_token in zip(prompts, negative_prompts, src_shifted_image_tokens, tgt_masked_image_tokens):
                if neg_prompt == '':
                    text = prompt
                    uncond_p = 1
                else:
                    text = (neg_prompt,)
                    uncond_p = 0
                neg_whole_tokens = self.tokenizer.encode_mlm_editing(
                    *text,
                    src_masked_shifted_image_tokens=src_shifted_image_token,
                    tgt_masked_shifted_image_tokens=tgt_masked_image_token,
                    src_image_token_len=image_token_len,
                    tgt_image_token_len=image_token_len,
                    max_text_token_length=self.arrange.s_text_maxlen,
                    max_total_token_length=self.arrange.s_text_maxlen + 2 * image_token_len,
                    instruction_uncond_p=uncond_p,
                    return_labels=False,
                    **kwargs,
                )[:-1]  # :-1 remove the eos token
                batch_uncond_whole_tokens.append(neg_whole_tokens)
            batch_uncond_whole_tokens = torch.stack(batch_uncond_whole_tokens, dim=0)
            batch_uncond_whole_tokens = batch_uncond_whole_tokens.to(device)
        else:
            batch_uncond_whole_tokens = None

        return batch_whole_tokens, batch_uncond_whole_tokens

    def encode_image(self, image, height, width, device, image_tokens=None):
        if image_tokens is None:
            image = self.image_processor.preprocess(image, height, width)

            vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                image_tokens = self.vae.vq_encode(image.to(device))
        else:
            image_tokens = image_tokens.to(device).unsqueeze(0)
        return image_tokens     # [bs, height, width]

    def prepare_attention_mask(self, whole_tokens, uncond_whole_tokens, image_ranges, negative_prompt, device):
        attention_mask = create_attention_mask_general(
            whole_tokens,
            self.arrange.pad_id,
            image_ranges,
            mask_pad=True,
            return_inverse_mask=True,
        )
        if self.do_classifier_free_guidance:
            if all([x == '' for x in negative_prompt]):
                attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            elif all([x != '' for x in negative_prompt]):
                negative_attention_mask = create_attention_mask_general(
                    uncond_whole_tokens,
                    self.arrange.pad_id,
                    image_ranges,
                    mask_pad=True,
                    return_inverse_mask=True,
                )
                attention_mask = torch.cat([negative_attention_mask, attention_mask], dim=0)
            else:
                raise ValueError(f"Negative prompt should be all empty or all non-empty, got {negative_prompt}")
        attention_mask = attention_mask.to(device)
        return attention_mask
