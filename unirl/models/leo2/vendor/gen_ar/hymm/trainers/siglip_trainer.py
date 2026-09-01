from transformers import AutoProcessor
import transformers
from hymm.trainers.base_trainer import BaseTrainer
from hymm.models.visual_encoders import build_model
from hymm.data_kits.text_image_siglip_loader import SiglipTextImageArrowStream

from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex, IndexBatchSampler
from torch.utils.data import DataLoader
from ..utils.torch_utils import set_worker_seed_builder
from ..utils.helpers import as_tuple
from .helpers import dynamic_values_wrapper

class SiglipTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.vision_encoder_max_num_patches = args.vision_encoder_max_num_patches
        self.text_encoder_max_length = args.text_encoder_max_length

    def build_evaluator(self):
        pass
    
    def build_model(self, dtype=None, device=None):
        model, model_config = build_model(self.args, self.logger, self.rank, self.world_size, dtype=dtype, device=device)
        return model, model_config

    def build_extra_model(self):
        args = self.args
        processor = AutoProcessor.from_pretrained(args.processor_path)
        self.image_processor = processor.image_processor
        self.tokenizer = processor.tokenizer

    def build_dataloader(self):
        args = self.args
        self.dataset = SiglipTextImageArrowStream(
            args=args,
            training_image_size=args.training_image_size,
            index_file=args.index_file,
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
                collate_fn=self.dataset.collate_fn,
                **dataloader_kwargs,
            )

    def prepare_model_inputs(self, batch, device):
        batch_size = len(batch["image"])
        vision_inputs = self.image_processor.preprocess(batch["image"], max_num_patches=self.vision_encoder_max_num_patches).to(device=device)

        text_inputs = self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=self.text_encoder_max_length, return_tensors="pt").to(device=device)

        return {
            "vision_inputs": vision_inputs,
            "text_inputs": text_inputs,
        }, batch_size, self.vision_encoder_max_num_patches
