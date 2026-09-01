from typing import Union, Dict

import torch
import torch.nn.functional as F

from .text2image_mlm_trainer import Text2ImageMLMTrainer
from ..ar.mask_schedulers import build_mask_ratio_generator
from ..ar.pipelines import DDPMPipeline
from ..diffusion import load_scheduler
from ..models import load_vae, TokenizerWrapper
from ..samplers.text2image_mar_sampler import Text2ImageMARSampler
from ..utils.file_utils import safe_file
from ..utils.torch_utils import PRECISION_TO_TYPE


# noinspection PyTypeChecker
class Text2ImageMARTrainer(Text2ImageMLMTrainer):
    def build_dataloader(self):
        self.tokenizer = TokenizerWrapper(self.args.tokenizer_name, self.logger)

        (
            self.dataset,
            self.data_sampler,
            self.dataloader,
        ) = self._build_dataloader(
            dataset_kwargs=dict(
                tokenizer=self.tokenizer,
                post_kwargs=dict(
                    patch_size=self.args.patch_size,
                )
            )
        )

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
        self.downsample_factor = self.vae.downsample_factor
        self.text_vocab_size = mconfig.padded_vocab_size

        # ====================== Maks Ratio Generator ======================
        self.mask_ratio_generator = build_mask_ratio_generator(
            schedule_type=args.get('train_schedule_type', 'arccos'),
            logger=self.logger,
        )

        # ====================== Arrangement ======================
        self.build_arrangement(dVAE=False)

    def get_sampler(self):
        self.diff_scheduler = load_scheduler(self.args)
        self.diff_pipeline = DDPMPipeline(
            model=self.model.diffloss.net,
            scheduler=self.diff_scheduler,
        )
        self.diff_pipeline.set_progress_bar_config(disable=True)

        return Text2ImageMARSampler(
            self.args,
            model_dict=dict(vae=self.vae,
                            model=self.model_engine.module,
                            model_settings=self.model_settings,
                            tokenizer=self.tokenizer,
                            arrange=self.arrange,
                            pipeline=self.diff_pipeline,
                            ),
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            logger=self.val_logger,
        )

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

        image = batch["image"].to(device)
        vae_autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            vae_encode_result = self.vae.encode(image)
            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            else:
                latents = vae_encode_result.latent_dist.sample()
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                latents.sub_(self.vae.config.shift_factor) #.mul_(self.vae.config.scaling_factor)
            if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
                latents.mul_(self.vae.config.scaling_factor)

        # Get mask
        mask_ratio = self.mask_ratio_generator(shape=(bs,))
        mask = torch.rand(size=(bs, tk_height * tk_width)) < mask_ratio.view(bs, 1)

        slice_s_image_range = slice(arrange.s_image_range[0], arrange.s_image_range[0] + tk_height * tk_width)
        imgs_start_pos = torch.tensor([arrange.s_image_range[0]] * bs, device=device)
        # Prepare whole labels
        whole_labels[:, slice_s_image_range] = arrange.ignore_id
        whole_labels = whole_labels.to(torch.long)

        # ===================================== Attention mask ==================================
        # Create attention mask: [bs, 1, seqlen, seqlen]
        attention_mask = self.get_attention_mask(whole_tokens)
        # ===================================== Build RoPE ==================================
        freqs_cos, freqs_sin = self.get_rope(tk_height, tk_width, device)

        # ===================================== Pack model kwargs ==================================
        model_kwargs = dict(
            idx=whole_tokens,               # [b, n], int32
            target=whole_labels,            # [b, n], int64
            attention_mask=attention_mask,  # [b, 1, n, n], float32
            freqs_cos=freqs_cos,            # [n, head_dim]
            freqs_sin=freqs_sin,            # [n, head_dim]
            imgs=latents,                   # [b, c, h*p, w*p], float32
            imgs_pos=imgs_start_pos,        # [b], int64
            imgs_mask=mask.to(device),      # [b, hw], bool
        )
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < args.save_n_training_data and self.rank < 8:
            check_data_path = self.exp_dir / f"training_samples/data_batch{cur_step}_rank{self.rank}.pt"
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            torch.save(model_kwargs, safe_file(check_data_path))

        # n_tokens is defined as the number of tokens participating in the loss calculation
        n_tokens = (whole_labels != args.ignore_index).sum().item() + (tk_height * tk_width)
        return model_kwargs, bs, n_tokens, (tk_height, tk_width), latents, mask

    @torch.no_grad()
    def reconstruct(self, tk_height, tk_width, latents=None, logits=None, mask=None, device=None, **kwargs):
        """
        Parameters
        ----------
        tk_height
        tk_width
        latents: torch.Tensor
            [bs, vae_ch, tk_height * patch_size, tk_width * patch_size]
        logits: torch.Tensor
            [bs, seqlen, n_embed]
        mask: torch.Tensor
            [bs, seqlen]
        device
        kwargs
        """
        image_token_lst = []
        height = tk_height * self.downsample_factor * self.model_settings.patch_size
        width = tk_width * self.downsample_factor * self.model_settings.patch_size

        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        target_dtype = PRECISION_TO_TYPE[self.args.autocast_dtype]

        def denormalize(x):
            return (x / 2 + 0.5).clamp(0, 1).cpu()

        def decode(inputs, denorm=True):
            if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
                inputs = inputs / self.vae.config.scaling_factor
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                inputs = inputs + self.vae.config.shift_factor
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                outputs = self.vae.decode(inputs.to(device), return_dict=False)[0]
            if denorm:
                outputs = denormalize(outputs)
            return outputs

        if latents is not None:
            # Decoding to pixel space
            _x = decode(latents, denorm=False)
            image_token_lst.append(denormalize(_x))

            if mask is not None:
                # Decoding to pixel space with mask tokens
                mask_ = mask.view(-1, tk_height, tk_width).unsqueeze(1).float()
                _x2 = _x * (1 - F.interpolate(mask_, (height, width)).to(_x.device))     # Masked area is 0 (gray)
                image_token_lst.append(denormalize(_x2))

        if logits is not None:
            # Decoding predicted tokens to pixel space
            logits_input = logits[mask] if mask is not None else logits

            with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=target_dtype != torch.float32):
                logits_pred = self.diff_pipeline(
                    cond=logits_input,
                    num_inference_steps=self.args.diff_infer_steps,
                    guidance_scale=1.0,
                )[0]    # [?, ch],     ch = vae_ch * patch_size ** 2
            if mask is not None:
                img_seqs = self.model.patchify(latents)[0]  # [bs, seqlen, ch]
                img_seqs[mask] = logits_pred
            else:
                img_seqs = logits_pred
            demasked_latents = self.model.unpatchify(img_seqs, tk_height, tk_width)
            image_token_lst.append(decode(demasked_latents))

        return torch.cat(image_token_lst, dim=0)
