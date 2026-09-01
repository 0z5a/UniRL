import os
import torch
from typing import Dict, Union

from einops import rearrange

from ..models import load_vae
from ..utils.file_utils import safe_dir
from .text2image_ar_trainer import Text2ImageARTrainer
from ..utils.torch_utils import PRECISION_TO_TYPE


# difference with Text2ImageARTrainer: Text2ImageARTrainer2 use pre-extracted token for training
class Text2ImageARTrainer2(Text2ImageARTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def build_extra_model(self):
        args = self.args
        use_pre_extracted_token = self.args.get("use_pre_extracted_token", False)
        if not use_pre_extracted_token:
            self.vae = load_vae(
                args.vae_type,
                args.vae_precision,
                device=self.device,
                logger=self.logger,
            )
            self.image_token_offset = args.image_token_offset
            self.logger.info(f"Image token offset: {self.image_token_offset}")

    def prepare_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(batch, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))

        # Online image tokenization
        use_pre_extracted_token = self.args.get("use_pre_extracted_token", False)
        if not use_pre_extracted_token:
            image = batch["image"].to(device)
            vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                    image_tokens = self.vae.vq_encode(image).long().detach()
            image_tokens = image_tokens + self.image_token_offset
            if len(image_tokens.shape) > 2:
                image_tokens = rearrange(image_tokens, "b h w -> b (h w)")

            # text: 256 + 1, image: 1024, total: 1281, 1281 - 1 = 1280
            tokens = batch["tokens"].contiguous().to(device)
            for i in range(tokens.shape[0]):
                tokens[i, tokens[i] == self.dataset.tokenizer.special_token_map["<img>"]] = image_tokens[i]
            
            target_token = tokens.clone()
            target_token[target_token == self.dataset.tokenizer.special_token_map["<pad>"]] = -100
            target_token[target_token == self.dataset.tokenizer.special_token_map["<cfg>"]] = -100
            target_token[target_token == self.dataset.tokenizer.special_token_map["<iw>"]] = -100
            target_token[target_token == self.dataset.tokenizer.special_token_map["<ih>"]] = -100
        else:
            # text: 256 + 1, image: 1024, total: 1281, 1281 - 1 = 1280
            tokens = batch["tokens"]
            target_token = batch["target_token"]
        
        tokens = tokens[:, :-1].contiguous().to(device)
        target_token = target_token[:, 1:].contiguous().to(device)
        text_loss_mask = batch["text_loss_mask"][:, 1:].contiguous().to(device)
        image_loss_mask = batch["image_loss_mask"][:, 1:].contiguous().to(device)

        # ===================================== Pack model kwargs ==================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 1280]
            target=target_token,  # [b, 1280]
            text_loss_mask=text_loss_mask,  # [b, 1280]
            image_loss_mask=image_loss_mask,  # [b, 1280]
        )
        if self.args.add_iw_ih_token:
            assert "iw_ih_scatter_index" in batch and "iw_ih_scatter_src" in batch, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"
            model_intput_kwargs.update({
                "iw_ih_scatter_index": batch["iw_ih_scatter_index"].to(device),  # [b, 2]
                "iw_ih_scatter_src": batch["iw_ih_scatter_src"].to(device),  # [b, 2]
            })

        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        return model_intput_kwargs, batch_size, n_tokens
