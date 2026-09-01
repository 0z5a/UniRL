import time
from collections import OrderedDict
from typing import Union, Dict

import torch
import torch.nn.functional as F
from index_kits.sampler import DistributedSamplerWithStartIndex
from torch.utils.data import DataLoader
from torchvision.utils import save_image as save_grid_image


from .base_trainer import BaseTrainer
from ..ar.mask_schedulers import build_mask_ratio_generator, get_mask_code
from ..constants import SCORE_METRICS, SAMPLE_METRICS
from ..data_kits.csv_dataset import LabelDataset
from ..data_kits.imagenet import ImageNetDataset
from ..models import load_vae
from ..samplers.text2image_mlm_sampler import Text2ImageMLMSampler
from ..utils.file_utils import (
    safe_file,
    safe_json,
)
from ..utils.helpers import to_2tuple
from ..utils.torch_utils import (
    set_worker_seed_builder,
    PRECISION_TO_TYPE,
)


# noinspection PyTypeChecker
class Label2ImageTrainerV2(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def build_extra_model(self):
        args = self.args
        mconfig = self.model_settings

        # ====================== Image Tokenizer ======================
        self.vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=self.device,
            logger=self.logger,
        )
        self.codebook_size = self.vae.codebook_size
        assert self.codebook_size == mconfig.media_vocab_size, (
            f"VQGAN codebook size {self.codebook_size} does not match media vocab size {mconfig.media_vocab_size}"
        )

        # ================= Training schedule ===================
        self.mask_ratio_generator = build_mask_ratio_generator(
            schedule_type=args.get('train_schedule_type', 'arccos'),
            logger=self.logger,
        )

        # ====================== Others ======================
        token_ranges = OrderedDict()
        if mconfig.vocab_image_first:
            self.image_vocab_range = (0, mconfig.media_vocab_size)
            self.mask_id = mconfig.media_vocab_size
            self.label_vocab_range = (mconfig.media_vocab_size + 1,
                                      mconfig.media_vocab_size + 1 + mconfig.padded_vocab_size)
            token_ranges['image tokens'] = self.image_vocab_range
            token_ranges['mask token'] = self.mask_id
            token_ranges['label tokens'] = self.label_vocab_range
        else:
            self.label_vocab_range = (0, mconfig.padded_vocab_size)
            self.image_vocab_range = (mconfig.padded_vocab_size,
                                      mconfig.padded_vocab_size + mconfig.media_vocab_size)
            self.mask_id = mconfig.padded_vocab_size + mconfig.media_vocab_size
            token_ranges['label tokens'] = self.label_vocab_range
            token_ranges['image tokens'] = self.image_vocab_range
            token_ranges['mask token'] = self.mask_id
        self.uncond_id = mconfig.media_vocab_size + 1 + mconfig.padded_vocab_size
        token_ranges['uncond token'] = self.uncond_id

        image_size = to_2tuple(args.image_size)
        self.tk_height = image_size[0] // self.vae.downsample_factor
        self.tk_width = image_size[1] // self.vae.downsample_factor
        self.image_seq_len = self.tk_height * self.tk_width

        self.logger.info(f"Token vocabulary:")
        for key, rng in token_ranges.items():
            if isinstance(rng, int):
                self.logger.info(f"  {key}: {rng} ~ {rng + 1} (1)")
            else:
                self.logger.info(f"  {key}: {rng[0]} ~ {rng[1]} ({rng[1] - rng[0]})")
        self.logger.info(f"Token sequence:")
        self.logger.info(f"  Class tokens: 0 ~ 1 (1)")
        self.logger.info(f"  Image tokens: 1 ~ approx {1 + self.image_seq_len} (approx {self.image_seq_len})")

    def build_evaluator(self):
        args = self.args
        eval_kwargs = dict(model_dict=dict(vae=self.vae,
                                           model=self.model_engine.module,
                                           model_settings=self.model_settings),
                           rank=self.rank,
                           world_size=self.world_size,
                           device=self.device,
                           logger=self.val_logger,
                           )

        metric_names = [metric.split('@')[0] for metric in args.validation_metrics]

        recon_metrics_simple = ["recon"]
        if recon_metrics_simple:
            self.recon_evaluator = True
            self.logger.info(f"Enable reconstruction evaluator: {self.recon_evaluator.__class__.__name__}")
            self.recon_metrics = recon_metrics_simple
            self.logger.info(f"  Metrics: {self.recon_metrics}")
        else:
            self.recon_evaluator = None

        sample_metrics_simple = set(metric_names) & SAMPLE_METRICS
        if sample_metrics_simple:
            self.sample_evaluator = Text2ImageMLMSampler(args, **eval_kwargs)
            self.logger.info(f"Enable sample evaluator: {self.sample_evaluator.__class__.__name__}")
            self.sample_metrics = sample_metrics_simple
            self.logger.info(f"  Metrics: {self.sample_metrics}")

        score_metrics_simple = set(metric_names) & SCORE_METRICS
        if score_metrics_simple:
            self.score_evaluator = self.sample_evaluator
            self.logger.info(f"Enable score evaluator: {self.score_evaluator.__class__.__name__}")
            self.score_metrics = sorted([
                metric for metric in args.validation_metrics if metric.split('@')[0] in score_metrics_simple
            ])
            self.logger.info(f"  Metrics: {self.score_metrics}")

    def build_dataloader(self):
        args = self.args

        # ImageNet dataset
        self.dataset = ImageNetDataset(image_size=args.image_size,
                                       index_file=args.index_file,
                                       logger=self.logger)
        # Build sampler and data loader
        dataloader_kwargs = dict(
            **args.dataloader_params, worker_init_fn=set_worker_seed_builder(self.rank)
        )
        self.data_sampler = DistributedSamplerWithStartIndex(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True if args.shuffle_by == 'sampler' else False,
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

    @torch.no_grad()
    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        args = self.args
        _ = batch["kwargs"]
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        bs, _, height, width = image.shape

        # ------------------------------------------------------------
        # 1. Tokenization
        vae_autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            image_tokens = self.vae.vq_encode(image)
            image_tokens = image_tokens.reshape(bs, -1)
        shifted_image_tokens = (image_tokens + self.image_vocab_range[0]).flatten(1)

        # Mask the encoded tokens
        masked_image_tokens, mask = get_mask_code(generator=self.mask_ratio_generator,
                                                  code=shifted_image_tokens,
                                                  mask_id=self.mask_id,
                                                  )

        # Drop xx% of the condition for cfg
        cls_token = label + self.label_vocab_range[0]  # Shift the class token by the amount of codebook
        drop_label = torch.empty(label.shape).uniform_(0, 1) < self.args.uncond_p
        cls_token[drop_label] = self.uncond_id  # Drop condition
        cls_token_expand = cls_token.unsqueeze(1)

        x = torch.cat([masked_image_tokens, cls_token_expand], dim=-1)  # concat visual tokens and class tokens
        if (loss_predict := args.get('loss_predict', 'unknown_known')) == 'unknown_known':
            # This strategy from https://github.com/valeoai/Maskgit-pytorch
            target = torch.cat([shifted_image_tokens, torch.full_like(cls_token_expand, args.ignore_index)], dim=-1)
        elif loss_predict == 'unknown':
            target_tokens = torch.where(mask.to(device), shifted_image_tokens, args.ignore_index)
            target = torch.cat([target_tokens, torch.full_like(cls_token_expand, args.ignore_index)], dim=-1)
        else:
            raise ValueError(f"Not supported loss_predict: {loss_predict}")

        # ===================================== Pack model kwargs ==================================
        model_kwargs = dict(
            x=x,            # [b, n], int64
            target=target,  # [b, n], int64
        )
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            check_data_path = self.exp_dir / f"training_samples/data_batch{cur_step}_rank{self.rank}.pt"
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            torch.save(model_kwargs, safe_file(check_data_path))

        bs, n_tokens = shifted_image_tokens.shape
        token_height = height // self.vae.downsample_factor
        token_width = width // self.vae.downsample_factor
        # assert n_tokens == token_height * token_width
        return model_kwargs, bs, n_tokens, (token_height, token_width), image_tokens, mask

    def train_step(self, batch):
        start1 = time.time()
        (
            model_kwargs,
            batch_size,
            n_tokens,
            (tk_height, tk_width),
            image_tokens,
            mask,
        ) = self.prepare_model_inputs(batch, self.device)
        duration1 = time.time() - start1

        start2 = time.time()
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            output_dict = self.model_engine(**model_kwargs)
        duration2 = time.time() - start2

        times = {
            "preprocess": duration1,
            "forward": duration2,
        }

        output_dict["tk_height"] = tk_height
        output_dict["tk_width"] = tk_width
        output_dict["tokens"] = image_tokens
        output_dict["mask"] = mask
        return output_dict, batch_size, n_tokens, times

    def eval_step(self, results_dict=None):
        args = self.args
        val_summary_events = []

        image_size = to_2tuple(args.image_size)
        batch_size = args.eval_batch_size
        size = f"{image_size[0]}x{image_size[1]}"

        kwargs = dict(
            infer_steps=args.infer_steps,
            guidance_scale=args.guidance_scale,
            pipeline_kwargs=args.get("pipeline_kwargs", {}),
        )

        if self.recon_evaluator is not None:
            self.val_logger.info(f"---------- Reconstruction Evaluator ----------")
            save_dir = self.exp_dir / f"reconstruction/{self.ss.update_steps:07d}_recon_{size}"
            save_path = str(save_dir / f"{self.rank:05d}.png")
            demasked_tokens = torch.softmax(
                results_dict['logits'][..., slice(*self.image_vocab_range)], dim=-1
            ).max(-1)[1][:, :-1]
            gen_sample = self.reconstruct(tk_height=results_dict['tk_height'],
                                          tk_width=results_dict['tk_width'],
                                          tokens=results_dict['tokens'][:8],
                                          demasked_tokens=demasked_tokens[:8],
                                          mask=results_dict['mask'][:8])
            save_grid_image(gen_sample, safe_file(save_path))
            self.logger.info(f"Save reconstruction samples to {save_dir}")

        if self.sample_evaluator is not None:
            self.val_logger.info(f"---------- Sample Evaluator ----------")
            suffix = self.sample_evaluator.get_sample_dir_suffix(testset="imagenet_1k",
                                                                 image_size=image_size,
                                                                 load_key="module",
                                                                 )
            save_template = str(self.exp_dir / "samples" / f"{self.ss.update_steps:07d}_{suffix}" / "{}_{{}}.png")
            dataset = LabelDataset(labels=list(range(0, self.model_settings.padded_vocab_size,
                                                     self.model_settings.padded_vocab_size // 100)),
                                   save_template=save_template)
            dataloader = self.sample_evaluator.build_sample_dataloader(
                data_source=dataset,
                batch_size=batch_size,
            )
            self.sample_evaluator.batch_sample(
                dataloader=dataloader,
                size=image_size,
                **kwargs,
            )

        if self.score_evaluator is not None:
            self.val_logger.info(f"---------- Score Evaluator ----------")
            save_template = str(self.exp_dir / "evaluation" / f"{self.ss.update_steps:07d}_score_{size}.json")

            self.score_evaluator.initialize_scores(self.score_metrics)
            self.score_evaluator.load_score_models(min(image_size))
            results = self.score_evaluator.eval(
                image_size=image_size,
                batch_size=batch_size,
                save_path=save_template,
                extra_save_info=safe_json(kwargs),
                **kwargs,
            )
            # Release score models to save GPU memory.
            self.score_evaluator.release_score_models(min(image_size))
            for item in results:
                val_summary_events.append(
                    (f"Val/Steps/{item['metric']}_{size}", item["value"], self.ss.update_steps)
                )

        if self.model_engine.monitor.enabled and self.rank == 0 and len(val_summary_events) > 0:
            self.model_engine.monitor.write_events(val_summary_events)

    @torch.no_grad()
    def reconstruct(self, tk_height, tk_width, tokens=None, demasked_tokens=None, mask=None):
        """ For visualization.

        Parameters
        ----------
        tk_height: int
            The height of the tokenized image.
        tk_width: int
            The width of the tokenized image.
        tokens: torch.Tensor, optional
            The original tokens.
        demasked_tokens: torch.Tensor, optional
            The tokens after model prediction on masked tokens.
        mask: torch.Tensor, optional
            The mask for the tokens.
        """
        image_token_lst = []
        height = tk_height * self.vae.downsample_factor
        width = tk_width * self.vae.downsample_factor

        def denormalize(x):
            return (x / 2 + 0.5).clamp(0, 1)

        if tokens is not None:
            image_tokens = tokens.view(-1, tk_height, tk_width)
            # Decoding to pixel space
            _x = self.vae.vq_decode(image_tokens)
            image_token_lst.append(denormalize(_x))

            if mask is not None:
                # Decoding to pixel space with mask tokens
                mask = mask.view(-1, 1, tk_height, tk_width).float()
                _x2 = _x * (1 - F.interpolate(mask, (height, width)).to(_x.device))     # Masked area is 0 (gray)
                image_token_lst.append(denormalize(_x2))

        if demasked_tokens is not None:
            # Decoding predicted tokens to pixel space
            demasked_tokens = demasked_tokens.view(-1, tk_height, tk_width)
            _x3 = self.vae.vq_decode(demasked_tokens)
            image_token_lst.append(denormalize(_x3))

        return torch.cat(image_token_lst, dim=0)
