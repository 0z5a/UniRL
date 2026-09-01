import os
from typing import Union, Dict
import torch
from hymm.trainers.textvisual2audio_trainer import VisualText2Audio


class VisualText2AudioMultiHeadTrainer(VisualText2Audio):
    def __init__(self, args):
        super().__init__(args)

    def prepare_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            check_data_path = os.path.join(self.exp_dir, f"data_batch{cur_step}_rank{self.rank}.pt")
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            torch.save(batch, check_data_path)

        clip_embedding = batch["clip_embedding"].to(device)
        t5_embedding = batch["t5_embedding"].to(device)
        input_token = batch["tokens"][:, :-1].contiguous().to(device)
        audio_tokens = batch["audio_tokens"].to(device)
        target_token = batch["target"].to(device)

        # ===================================== Pack model kwargs ==================================
        model_kwargs = dict(
            idx=input_token,  # [b, l-1]
            audio_token=audio_tokens, 
            clip_feat=clip_embedding,  # [b, xxx, 2048]
            t5_feat=t5_embedding,  # [b, xxx, 2048]
            target=target_token,  # [b, l-1]
        )

        batch_size = input_token.shape[0]
        n_tokens = input_token.shape[1]
        return model_kwargs, batch_size, n_tokens

    def build_extra_model(self):
        """leave tokenizer to be implemented by subclass,
        since we may neeed to do experiments on different modalities with different tokenizers
        """
        pass
