import time
from typing import Union, Dict

import torch
import torch.nn.functional as F
from index_kits.sampler import BlockDistributedSampler
from torch.utils.data import DataLoader
from torchvision.utils import save_image as save_grid_image

from .base_trainer import BaseTrainer
from ..data_kits.instruction_tuning_mlm_loader import InstructionTuningMaskArrowStream
from ..models import load_vae, Arrangement, TokenizerWrapper
from ..utils.file_utils import safe_file, dump_configs
from ..utils.helpers import to_2tuple, default
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE
from ..utils.torch_distributions import categorical_sample


# noinspection PyTypeChecker
class InstructionTuningMLMTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.rope_cache = {}

    def _build_dataloader(self, dataset_kwargs=None, sampler_kwargs=None, dataloader_kwargs=None):
        args = self.args
        dataset = InstructionTuningMaskArrowStream(
            args=args,
            index_file=args.index_file,
            image_token_length=args.image_token_length,
            image_token_offset=args.image_token_offset,
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
                                collate_fn=dataset.collate_fn,
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

        # ====================== Arrangement ======================
        self.build_arrangement(dVAE=True)

    def build_arrangement(self, dVAE=True):
        args = self.args
        # ====================== Token Specification ======================
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
            s_text_range=(None, args.text_token_length - 5),
            s_text_maxlen=args.text_token_length,
            s_image_range=(args.text_token_length - 4, None),
            s_image_maxlen=-1,
            sequence=["<pad>", "<bos>", "<text>", "<boi>", "<image>", "<eoi>", "<boi>", "<image>", "<eoi>", "<eos>"],
            ignore_id=args.ignore_index,
        )
        if self.rank == 0:
            dump_configs(self.arrange, self.exp_dir / "arrangement.yaml")
        self.logger.info(f"\n{self.arrange}")

    def build_evaluator(self):
        recon_metrics_simple = ["recon"]
        if recon_metrics_simple:
            self.recon_evaluator = True
            self.logger.info(f"Enable reconstruction evaluator: {self.recon_evaluator.__class__.__name__}")
            self.recon_metrics = recon_metrics_simple
            self.logger.info(f"  Metrics: {self.recon_metrics}")
        else:
            self.recon_evaluator = None

    @torch.no_grad()
    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        args = self.args

        # ===================================== Tokenization ==================================
        # [<pad>, <text>, <boi>, <img>, <eoi>, <boi>, <img>, <eoi>, <eos>]
        whole_tokens = batch["whole_tokens"][:, :-1].contiguous().to(device)
        whole_labels = batch["whole_labels"][:, 1:].contiguous().to(device)
        tgt_image_loss_mask = batch["tgt_image_loss_mask"][:, 1:].contiguous().to(device)
        text_loss_mask = batch["text_loss_mask"][:, 1:].contiguous().to(device) \
            if batch['text_loss_mask'] is not None else None
        attention_mask = batch["attention_mask"].to(device)
        freqs_cos = batch["freqs_cos"].to(device)
        freqs_sin = batch["freqs_sin"].to(device)
        bsz, _ = whole_tokens.shape

        # ===================================== Pack model kwargs ==================================
        model_kwargs = dict(
            idx=whole_tokens,               # [b, n], int32
            target=whole_labels,            # [b, n], int64
            image_loss_mask=tgt_image_loss_mask,  # [b, n], float32
            text_loss_mask=text_loss_mask,  # [b, n], float32
            image_loss_weight=3.0 if not args.loss_predict.startswith('target_only') else 1.0,
            only_image_loss=args.loss_predict.startswith('target_only'),
            attention_mask=attention_mask,  # [b, 1, n, n], float32
            freqs_cos=freqs_cos,            # [b, n, head_dim]
            freqs_sin=freqs_sin,            # [b, n, head_dim]
        )
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < args.save_n_training_data and self.rank < 8:
            check_data_path = self.exp_dir / f"training_samples/data_batch{cur_step}_rank{self.rank}.pt"
            save_model_kwargs = dict(
                src_image_tokens_shape_wh=batch["src_image_tokens_shape_wh"],
                tgt_image_tokens_shape_wh=batch["tgt_image_tokens_shape_wh"],
                **model_kwargs
            )
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            torch.save(save_model_kwargs, safe_file(check_data_path))

        # n_tokens is defined as the number of tokens participating in the loss calculation
        n_tokens = (whole_labels != args.ignore_index).sum().item()
        return (
            model_kwargs, bsz, n_tokens,
            batch["src_image_tokens_shape_wh"], batch["tgt_image_tokens_shape_wh"],
            batch["src_image_tokens"], batch["tgt_image_tokens"], batch["mask"], batch["tgt_mask"]
        )

    def train_step(self, batch):
        start1 = time.time()
        (
            model_kwargs,
            batch_size,
            n_tokens,
            src_tk_wh,
            tgt_tk_wh,
            src_image_tokens,
            tgt_image_tokens,
            user_src_masks, tgt_masks,
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

        output_dict["src_tk_wh"] = src_tk_wh
        output_dict["tgt_tk_wh"] = tgt_tk_wh
        output_dict["src_tokens"] = src_image_tokens
        output_dict["tgt_tokens"] = tgt_image_tokens
        output_dict["user_src_masks"] = user_src_masks
        output_dict["tgt_masks"] = tgt_masks
        return output_dict, batch_size, n_tokens, times

    def eval_step(self, results_dict=None):
        args = self.args
        # Performing validation on multiple image sizes.
        val_summary_events = []

        image_size = to_2tuple(args.image_size)
        size = f"{image_size[0]}x{image_size[1]}"

        if self.recon_evaluator is not None:
            self.val_logger.info(f"---------- Reconstruction Evaluator ----------")
            save_dir = self.exp_dir / f"reconstruction/{self.ss.update_steps:07d}_recon_{size}"
            save_path = str(save_dir / f"{self.rank:05d}.png")
            nrow = min(8, results_dict['logits'].size(0))
            gen_sample = self.reconstruct(results_dict['src_tk_wh'],
                                          results_dict['tgt_tk_wh'],
                                          results_dict['src_tokens'][:nrow],
                                          results_dict['tgt_tokens'][:nrow],
                                          logits=results_dict['logits'][:nrow],
                                          user_src_masks=results_dict['user_src_masks'][:nrow],
                                          tgt_masks=results_dict['tgt_masks'][:nrow],
                                          device=self.device,
                                          )
            save_grid_image(gen_sample, safe_file(save_path), nrow=nrow)
            self.logger.info(f"Save reconstruction samples to {save_dir}")

        if self.model_engine.monitor.enabled and self.rank == 0 and len(val_summary_events) > 0:
            self.model_engine.monitor.write_events(val_summary_events)

    @torch.no_grad()
    def reconstruct(self, src_tk_wh, tgt_tk_wh, src_tokens=None, tgt_tokens=None,
                    logits=None, user_src_masks=None, tgt_masks=None, device=None):
        """ For visualization.

        Parameters
        ----------
        src_tk_wh: int
            The size of the source tokenized image.
        tgt_tk_wh: int
            The size of the target tokenized image.
        src_tokens: torch.Tensor, optional
            The original source tokens.
        tgt_tokens: torch.Tensor, optional
            The original target tokens.
        logits: torch.Tensor, optional
            The tokens after model prediction on masked tokens.
        user_src_masks: torch.Tensor, optional
            The user defined mask for the source tokens.
        tgt_masks: torch.Tensor, optional
            The mask for the target tokens.
        device: torch.device, optional
        """
        image_token_lst = []
        bsz = src_tk_wh.size(0)
        max_w = max(src_tk_wh[:, 0].max().item(), tgt_tk_wh[:, 0].max().item()) * self.downsample_factor
        max_h = max(src_tk_wh[:, 1].max().item(), tgt_tk_wh[:, 1].max().item()) * self.downsample_factor
        src = torch.zeros(bsz, 3, max_h, max_w)
        image_token_lst.append(src)
        masked_src = torch.zeros_like(src)
        image_token_lst.append(masked_src)
        tgt = torch.zeros_like(src)
        masked_tgt = torch.zeros_like(src)
        demasked_tgt = torch.zeros_like(src)
        image_token_lst.extend([tgt, masked_tgt, demasked_tgt])

        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]

        def denormalize(x):
            return (x / 2 + 0.5).clamp(0, 1).cpu()

        def decode(inputs):
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                outputs = self.vae.vq_decode(inputs.to(device))
            return outputs

        for i in range(bsz):
            src_tk_width, src_tk_height = src_tk_wh[i].tolist()
            tgt_tk_width, tgt_tk_height = tgt_tk_wh[i].tolist()
            src_height = src_tk_height * self.downsample_factor
            src_width = src_tk_width * self.downsample_factor
            tgt_height = tgt_tk_height * self.downsample_factor
            tgt_width = tgt_tk_width * self.downsample_factor

            src_image_tokens = src_tokens[i].view(1, src_tk_height, src_tk_width)
            _x = decode(src_image_tokens)
            src[i, :, :src_height, :src_width] = denormalize(_x)

            user_src_mask = user_src_masks[i]
            if user_src_mask is not None:
                user_src_mask = user_src_mask.view(1, src_tk_height, src_tk_width)
                _x2 = _x * (1 - F.interpolate(user_src_mask.unsqueeze(1).float(), (src_height, src_width)).to(_x.device))     # Masked area is 0 (gray)
                masked_src[i, :, :src_height, :src_width] = denormalize(_x2)

            tgt_image_tokens = tgt_tokens[i].view(1, tgt_tk_height, tgt_tk_width)
            _y = decode(tgt_image_tokens)
            tgt[i, :, :tgt_height, :tgt_width] = denormalize(_y)

            tgt_mask = tgt_masks[i].view(1, tgt_tk_height, tgt_tk_width)
            _y2 = _y * (1 - F.interpolate(tgt_mask.unsqueeze(1).float(), (tgt_height, tgt_width)).to(_y.device))     # Masked area is 0 (gray)
            masked_tgt[i, :, :tgt_height, :tgt_width] = denormalize(_y2)

            tgt_start = self.arrange.s_image_range[0] + src_tk_height * src_tk_width + 2
            rt_s_image_range = (tgt_start, tgt_start + tgt_tk_height * tgt_tk_width)
            demasked_tokens = categorical_sample(torch.softmax(
                logits[i, slice(*rt_s_image_range), slice(*self.arrange.image_id_range)],
                dim=-1
            ))
            demasked_tokens = demasked_tokens.view(-1, tgt_tk_height, tgt_tk_width)
            demasked_tokens = torch.where(
                tgt_mask.to(demasked_tokens.device), demasked_tokens, tgt_image_tokens.to(demasked_tokens.device))
            _y3 = decode(demasked_tokens)
            demasked_tgt[i, :, :tgt_height, :tgt_width] = denormalize(_y3)

        return torch.cat(image_token_lst, dim=0)
