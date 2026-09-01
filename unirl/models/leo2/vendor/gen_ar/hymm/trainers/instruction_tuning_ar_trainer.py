import os
import torch
from typing import Dict, Union

from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex, IndexBatchSampler
from torch.utils.data import DataLoader

from .helpers import dynamic_values_wrapper
from ..utils.helpers import as_tuple
from ..utils.torch_utils import set_worker_seed_builder
from ..utils.file_utils import safe_dir
from .base_trainer import BaseTrainer
from ..data_kits.instruction_tuning_ar_loader import InstructionTuningARArrowStream


class InstructionTuningARTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def build_extra_model(self):
        pass

    def build_dataloader(self):
        args = self.args
        self.dataset = InstructionTuningARArrowStream(
            args=args,
            index_file=args.index_file,
            image_token_length=args.image_token_length,
            image_token_offset=args.image_token_offset,
            text_token_length=args.text_token_length,
            tokenizer_name=args.tokenizer_name,
            multireso=args.multireso,
            index_kwargs=dict(
                batch_size=1 if args.mix_scale else self.micro_batch_size,
                world_size=1 if args.mix_scale else self.world_size,
                **args.index_kwargs,
            ),
            debug=False,
            logger=self.logger,
        )
        # Build sampler and data loader
        dataloader_kwargs = dict(
            **args.dataloader_params, worker_init_fn=set_worker_seed_builder(self.rank)
        )
        # Dynamic anchor size for mix-scale training.
        self.anchor_sizes = as_tuple(args.anchor_size)
        self.dynamic_anchor_size = dynamic_values_wrapper(self.anchor_sizes, self.anchor_sizes)
        if args.mix_scale:
            self.data_sampler = DistributedSamplerWithStartIndex(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=args.global_seed,
                drop_last=True,
            )
            dynamic_batch_size = dynamic_values_wrapper(self.anchor_sizes, self.args.mix_micro_batch_size)
            batch_sampler = IndexBatchSampler(
                self.dataset.index_manager, self.data_sampler, batch_size=dynamic_batch_size, drop_last=True
            )
            self.dataloader = DataLoader(self.dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
        else:
            if args.multireso:
                self.data_sampler = BlockDistributedSampler(
                    self.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    seed=args.global_seed,
                    drop_last=True,
                    align=self.micro_batch_size,
                )
            else:
                self.data_sampler = DistributedSamplerWithStartIndex(
                    self.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    seed=args.global_seed,
                    drop_last=True,
                )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.micro_batch_size,
                sampler=self.data_sampler,
                shuffle=False,
                drop_last=True,
                **dataloader_kwargs,
            )

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
        # text: 256 + 1, image: 1024 * 2, total: 2305, 2305 - 1 = 2304
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_token = batch["target_token"][:, 1:].contiguous().to(device)
        text_loss_mask = batch["text_loss_mask"][:, 1:].contiguous().to(device)
        src_image_loss_mask = batch["src_image_loss_mask"][:, 1:].contiguous().to(device)
        tgt_image_loss_mask = batch["tgt_image_loss_mask"][:, 1:].contiguous().to(device)
        # ===================================== Pack model kwargs ==================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 2304]
            target=target_token,  # [b, 2304]
            text_loss_mask=text_loss_mask,  # [b, 2304]
            image_loss_mask=tgt_image_loss_mask,  # [b, 2304]
            image_loss_weight=3.0,
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
