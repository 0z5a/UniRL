import time
from pathlib import Path
from typing import Union, Dict

import torch
import torch.nn.functional as F
from index_kits.sampler import BlockDistributedSampler
from torch.utils.data import DataLoader
from torchvision.utils import save_image as save_grid_image

from .base_trainer import BaseTrainer
from ..ar.mask_schedulers import (
    create_attention_mask_t2i,
    get_mask_code,
    build_mask_ratio_generator,
)
from ..constants import SCORE_METRICS, SAMPLE_METRICS
from ..data_kits.text_image_mlm_loader import TextMaskImageArrowStream
from ..models import load_vae, Arrangement, TokenizerWrapper
from ..models.basic.rope import get_mlm_rope
from ..samplers.text2image_mlm_sampler import Text2ImageMLMSampler
from ..utils.file_utils import safe_file, dump_configs
from ..utils.helpers import to_2tuple, default
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE
from ..utils.torch_distributions import categorical_sample
from hymm.models.autoencoders.emu2.vit_enc_dif_dec import EVACLIP_256x16x16_SDXL

# noinspection PyTypeChecker
class Text2ImageMLMTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.rope_cache = {}

    def _build_dataloader(self, dataset_kwargs=None, sampler_kwargs=None, dataloader_kwargs=None):
        args = self.args
        dataset = TextMaskImageArrowStream(
            args=args,
            index_file=args.index_file,
            training_image_size=args.image_size,
            image_token_length=-1,                                  # not used
            image_token_offset=args.image_token_offset,
            use_pre_extracted_token=args.use_pre_extracted_token,
            text_token_length=args.text_token_length,
            uncond_p=args.uncond_p,
            multireso=args.multireso,
            index_kwargs=dict(
                batch_size=self.micro_batch_size,
                world_size=self.world_size,
                **args.index_kwargs,
            ),
            logger=self.logger,
            **default(dataset_kwargs, {}),
        )
        # Build sampler and data loader
        data_sampler = BlockDistributedSampler(dataset,
                                               num_replicas=self.world_size,
                                               rank=self.rank,
                                               shuffle=args.shuffle_by == "sampler",
                                               seed=args.global_seed,
                                               drop_last=True,
                                               align=self.micro_batch_size,
                                               **default(sampler_kwargs, {}),
                                               )
        dataloader = DataLoader(dataset,
                                batch_size=self.micro_batch_size,
                                sampler=data_sampler,
                                shuffle=False,
                                drop_last=True,
                                worker_init_fn=set_worker_seed_builder(self.rank),
                                **args.dataloader_params,
                                **default(dataloader_kwargs, {}),
                                )
        return dataset, data_sampler, dataloader

    def build_dataloader(self):
        self.tokenizer = TokenizerWrapper(self.args.tokenizer_name, self.logger)

        (
            self.dataset,
            self.data_sampler,
            self.dataloader,
        ) = self._build_dataloader(dataset_kwargs=dict(tokenizer=self.tokenizer))

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
        assert self.vae.codebook_size == args.media_vocab_size, (
            f"VQGAN codebook size {self.vae.codebook_size} does not match media vocab size {args.media_vocab_size}"
        )
        self.downsample_factor = self.vae.downsample_factor
        self.media_vocab_size = self.vae.codebook_size
        self.text_vocab_size = mconfig.padded_vocab_size

        # ====================== Maks Ratio Generator ======================
        self.mask_ratio_generator = build_mask_ratio_generator(
            schedule_type=args.get('train_schedule_type', 'arccos'),
            logger=self.logger,
        )

        # ====================== Arrangement ======================
        self.build_arrangement(dVAE=True)

    def build_arrangement(self, dVAE=True):
        args = self.args
        # ====================== Token Specification ======================
        image_size = to_2tuple(args.image_size)
        image_id_range = (self.text_vocab_size, self.text_vocab_size + self.media_vocab_size) if dVAE else (-1, -1)
        self.arrange = Arrangement(
            text_id_range=(0, self.text_vocab_size),
            image_id_range=image_id_range,
            pad_id=self.tokenizer.special_token_map["<pad>"],
            img_id=self.tokenizer.special_token_map["<img>"],
            mask_id=self.tokenizer.special_token_map["<mask>"],
            uncond_id=self.tokenizer.special_token_map["<cfg>"],
            boi_id=self.tokenizer.special_token_map["<boi>"],
            eoi_id=self.tokenizer.special_token_map["<eoi>"],
            s_text_range=(None, args.text_token_length - 3),
            s_text_maxlen=args.text_token_length,
            s_image_range=(args.text_token_length - 2, None),
            s_image_maxlen=(image_size[0] * image_size[1]) // self.downsample_factor ** 2,
            sequence=["<pad>", "<bos>", "<text>", "<boi>", "<image>", "<eoi>", "<eos>"],
            ignore_id=args.ignore_index,
        )
        if self.rank == 0:
            dump_configs(self.arrange, self.exp_dir / "arrangement.yaml")
        self.logger.info(f"\n{self.arrange}")

    def get_sampler(self):
        return Text2ImageMLMSampler(
            self.args,
            model_dict=dict(vae=self.vae,
                            model=self.model_engine.module,
                            model_settings=self.model_settings,
                            tokenizer=self.tokenizer,
                            arrange=self.arrange,
                            ),
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            logger=self.val_logger,
        )

    def build_evaluator(self):
        args = self.args

        val_metrics = default(args.validation_metrics, [])
        metric_names = [metric.split('@')[0] for metric in val_metrics]

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
            self.sample_evaluator = self.get_sampler()
            self.logger.info(f"Enable sample evaluator: {self.sample_evaluator.__class__.__name__}")
            self.sample_metrics = sample_metrics_simple
            self.logger.info(f"  Metrics: {self.sample_metrics}")

        score_metrics_simple = set(metric_names) & SCORE_METRICS
        if score_metrics_simple:
            self.score_evaluator = self.sample_evaluator
            self.logger.info(f"Enable score evaluator: {self.score_evaluator.__class__.__name__}")
            self.score_metrics = sorted([
                metric for metric in val_metrics if metric.split('@')[0] in score_metrics_simple
            ])
            self.logger.info(f"  Metrics: {self.score_metrics}")

    def get_attention_mask(self, whole_tokens):
        return create_attention_mask_t2i(
            whole_tokens,
            self.arrange.pad_id,
            self.arrange.boi_id,
            self.arrange.eoi_id,
            mask_pad=True,
            return_inverse_mask=True,
            dtype=self.target_dtype,
        )

    def get_rope(self, height, width, device):
        if self.model_settings.rope_type in ["3d", "3d-interleave"]:
            if (height, width) in self.rope_cache:
                freqs_cos, freqs_sin = self.rope_cache[(height, width)]
            else:
                head_dim = self.model_settings.n_embd // self.model_settings.n_head
                text_dim = int(head_dim * self.model_settings.rotary_percentage)
                image_half_dim = (head_dim - text_dim) // 2
                assert text_dim + 2 * image_half_dim == head_dim, "RoPE dimension mismatch"
                rope_dim_list = [text_dim, image_half_dim, image_half_dim]
                rope_kwargs = dict(theta=self.args.get('rope_theta', 10000), use_real=True,
                                   interleave=self.model_settings.rope_type == "3d-interleave")
                freqs_cos, freqs_sin = get_mlm_rope(
                    rope_dim_list, height, width, self.arrange.s_text_maxlen, self.arrange.s_text_maxlen - 2,
                    device, **rope_kwargs)
                self.rope_cache[(height, width)] = (freqs_cos, freqs_sin)
        elif self.model_settings.rope_type == "default":
            freqs_cos, freqs_sin = None, None
        else:
            raise ValueError(f"Unknown RoPE type: {self.model_settings.rope_type}")
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        args = self.args
        arrange = self.arrange
        _ = batch["kwargs"]     # index, text, target_size, ...
        tk_width, tk_height = batch["image_tokens_shape_wh"][0].tolist()

        # ===================================== Tokenization ==================================
        # [<pad>, <text>, <boi>, <img>, <eoi>, <eos>]
        whole_tokens = batch["whole_tokens"][:, :-1].contiguous().to(device)
        whole_labels = batch["whole_labels"][:, 1:].contiguous().to(device)
        bs, _ = whole_tokens.shape

        if args.use_pre_extracted_token:
            image_tokens = batch["image_tokens"].flatten(1)            # unmasked image tokens
        else:
            image = batch["image"].to(device)
            vae_autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]
            with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                image_tokens = self.vae.vq_encode(image).flatten(1)

        shifted_image_tokens = (image_tokens + arrange.image_id_range[0]).to(device)
        # Apply mask to image token
        masked_shifted_image_tokens, mask = get_mask_code(
            generator=self.mask_ratio_generator,
            code=shifted_image_tokens,
            mask_id=arrange.mask_id,
        )

        # Prepare whole tokens
        slice_s_image_range = slice(arrange.s_image_range[0], arrange.s_image_range[0] + tk_height * tk_width)
        whole_tokens[:, slice_s_image_range] = masked_shifted_image_tokens

        # Prepare whole labels
        if "_masked" in args.loss_predict:
            whole_labels[:, slice_s_image_range] = torch.where(mask.to(device), shifted_image_tokens, arrange.ignore_id)
        else:
            whole_labels[:, slice_s_image_range] = masked_shifted_image_tokens
        whole_labels = whole_labels.to(torch.long)

        # ===================================== Loss mask ==================================
        if 'text' in args.loss_predict:
            image_loss_mask = torch.zeros_like(whole_tokens, dtype=torch.float32, device=device)
            image_loss_mask[whole_tokens == arrange.mask_id] = 1.0
            text_loss_mask = torch.zeros_like(whole_tokens, dtype=torch.float32, device=device)
            text_loss_mask[
                (whole_labels > arrange.text_id_range[0]) & (whole_labels < self.model_settings.vocab_size)
            ] = 1.0
        else:
            image_loss_mask = None
            text_loss_mask = None
        # ===================================== Attention mask ==================================
        # Create attention mask: [bs, 1, seqlen, seqlen]
        attention_mask = self.get_attention_mask(whole_tokens)
        # ===================================== Build RoPE ==================================
        freqs_cos, freqs_sin = self.get_rope(tk_height, tk_width, device)

        # ===================================== Pack model kwargs ==================================
        model_kwargs = dict(
            idx=whole_tokens,               # [b, n], int32
            target=whole_labels,            # [b, n], int64
            image_loss_mask=image_loss_mask,  # [b, n], float32
            text_loss_mask=text_loss_mask,  # [b, n], float32
            attention_mask=attention_mask,  # [b, 1, n, n], float32
            freqs_cos=freqs_cos,            # [n, head_dim]
            freqs_sin=freqs_sin,            # [n, head_dim]
        )
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < args.save_n_training_data and self.rank < 8:
            check_data_path = self.exp_dir / f"training_samples/data_batch{cur_step}_rank{self.rank}.pt"
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            torch.save(model_kwargs, safe_file(check_data_path))

        # n_tokens is defined as the number of tokens participating in the loss calculation
        n_tokens = (whole_labels != args.ignore_index).sum().item()
        return model_kwargs, bs, n_tokens, (tk_height, tk_width), image_tokens, mask

    def train_step(self, batch):
        if hasattr(self.vae, "set_logging_enabled"):
            self.should_log = (self.ss.update_steps % (self.args.log_every*10) == 0) or self.ss.update_steps == 0
            if self.should_log: self.logger.info(f"Set vae logging enabled to {self.should_log=}, at {self.ss.update_steps=}, controlled by {self.args.log_every=}")
            self.vae.set_logging_enabled(enabled=self.should_log)
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
        # Performing validation on multiple image sizes.
        val_summary_events = []

        if hasattr(self.vae, "set_logging_enabled"):
            self.vae.set_logging_enabled(enabled=True)
            self.logger.info(f"Set vae logging enabled to True in the beginning of eval_step")
        if self.recon_evaluator is not None:
            self.val_logger.info(f"---------- Reconstruction Evaluator ----------")
            train_image_size = to_2tuple(args.train_image_size)
            size = f"{train_image_size[0]}x{train_image_size[1]}"
            save_dir = self.exp_dir / f"reconstruction/{self.ss.update_steps:07d}_recon_{size}"
            save_path = str(save_dir / f"{self.rank:05d}.png")
            nrow = 8
            gen_sample = self.reconstruct(results_dict['tk_height'],
                                          results_dict['tk_width'],
                                          results_dict['tokens'][:nrow],
                                          logits=results_dict['logits'][:nrow],
                                          mask=results_dict['mask'][:nrow],
                                          random=True,
                                          device=self.device,
                                          )
            save_grid_image(gen_sample, safe_file(save_path), nrow=nrow)
            self.logger.info(f"Save reconstruction samples to {save_dir}")

        if self.sample_evaluator is not None:
            self.val_logger.info(f"---------- Sample Evaluator ----------")
            sample_image_size = to_2tuple(args.metric_image_size)
            suffix = self.sample_evaluator.get_sample_dir_suffix(testset=Path(args.csv).stem,
                                                                 image_size=sample_image_size,
                                                                 load_key="module",
                                                                 )
            save_template = str(self.exp_dir / "samples" / f"{self.ss.update_steps:07d}_{suffix}" / "{}_{{}}.png")
            dataloader = self.sample_evaluator.build_sample_dataloader(
                data_source=args.csv,
                save_template=save_template,
                batch_size=args.sample_batch_size,
            )
            self.sample_evaluator.batch_sample(
                dataloader=dataloader,
                size=sample_image_size,
            )

        if self.score_evaluator is not None:
            self.val_logger.info(f"---------- Score Evaluator ----------")
            metric_image_size = to_2tuple(args.metric_image_size)
            size = f"{metric_image_size[0]}x{metric_image_size[1]}"
            save_template = str(self.exp_dir / "evaluation" / f"{self.ss.update_steps:07d}_score_{size}.json")

            self.val_logger.info(f"Loading score models...")
            self.score_evaluator.initialize_scores(self.score_metrics)
            self.score_evaluator.load_score_models(min(metric_image_size))
            # image_save_base = str(self.exp_dir / "samples" / f"{self.ss.update_steps:07d}")
            image_save_base = (
                # self.score_evaluator.get_sample_save_dir(testset=None, image_size=image_size)
                str(self.exp_dir / "samples" / f"{self.ss.update_steps:07d}")
                if args.validation_metrics_save_image
                else None
            )
            results = self.score_evaluator.eval(
                image_size=metric_image_size,
                batch_size=args.metric_batch_size,
                save_path=save_template,
                extra_save_info=self.score_evaluator.get_infer_kwargs(),
                image_save_base=image_save_base
            )
            # Release score models to save GPU memory.
            self.val_logger.info(f"  Releasing score models...")
            self.score_evaluator.release_score_models(min(metric_image_size))
            for item in results:
                val_summary_events.append(
                    (f"Val/Steps/{item['metric']}_{size}", item["value"], self.ss.update_steps)
                )

        if self.model_engine.monitor.enabled and self.rank == 0 and len(val_summary_events) > 0:
            self.model_engine.monitor.write_events(val_summary_events)
        if hasattr(self.vae, "set_logging_enabled"):
            self.vae.set_logging_enabled(enabled=False)
            self.logger.info(f"Set vae logging enabled to False in the end of eval_step")

    @torch.no_grad()
    def reconstruct(self, tk_height, tk_width, tokens=None, logits=None, mask=None, random=False, device=None):
        """ For visualization.

        Parameters
        ----------
        tk_height: int
            The height of the tokenized image.
        tk_width: int
            The width of the tokenized image.
        tokens: torch.Tensor, optional
            The original tokens.
        logits: torch.Tensor, optional
            The tokens after model prediction on masked tokens.
        mask: torch.Tensor, optional
            The mask for the tokens.
        random: bool, optional
            Whether to add a line of filling random tokens in the masked area.
        device: torch.device, optional
        """
        image_token_lst = []
        height = tk_height * self.downsample_factor
        width = tk_width * self.downsample_factor

        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]

        def denormalize(x):
            return (x / 2 + 0.5).clamp(0, 1).cpu()

        def decode(inputs):
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                outputs = self.vae.vq_decode(inputs.to(device))
            return outputs

        if tokens is not None:
            image_tokens = tokens.view(-1, tk_height, tk_width)
            # Decoding to pixel space
            _x = decode(image_tokens)
            image_token_lst.append(denormalize(_x))

            if mask is not None:
                # Decoding to pixel space with mask tokens
                mask = mask.view(-1, tk_height, tk_width)
                _x2 = _x * (1 - F.interpolate(mask.unsqueeze(1).float(), (height, width)).to(_x.device))     # Masked area is 0 (gray)
                image_token_lst.append(denormalize(_x2))

                if random:
                    # Decoding to pixel space with random tokens in the masked area
                    random_tokens = torch.randint_like(
                        image_tokens, 0, self.media_vocab_size, dtype=torch.long, device=image_tokens.device)
                    random_tokens = torch.where(mask.to(image_tokens.device), random_tokens, image_tokens)
                    _x_rand = decode(random_tokens)
                    image_token_lst.append(denormalize(_x_rand))

        if logits is not None:
            rt_s_image_range = (self.arrange.s_image_range[0], self.arrange.s_image_range[0] + tk_height * tk_width)
            demasked_tokens = categorical_sample(torch.softmax(
                logits[:, slice(*rt_s_image_range), slice(*self.arrange.image_id_range)],
                dim=-1
            ))
            # Decoding predicted tokens to pixel space
            demasked_tokens = demasked_tokens.view(-1, tk_height, tk_width)
            if tokens is not None and mask is not None:
                image_tokens = tokens.view(-1, tk_height, tk_width)
                mask = mask.view(-1, tk_height, tk_width)
                demasked_tokens = torch.where(
                    mask.to(demasked_tokens.device), demasked_tokens, image_tokens.to(demasked_tokens.device))

            _x3 = decode(demasked_tokens)
            image_token_lst.append(denormalize(_x3))

        return torch.cat(image_token_lst, dim=0)
