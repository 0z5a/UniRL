from functools import partial

from .pretrain_pure_torch import MultimodalTrainer
from ..constants import LI_DIT_PROMPT_TEMPLATE
from ..core.data_provider import DatasetsProvider, save_first_training_samples
from ..core.extra_model_provider import (
    build_scalar_state,
    build_vae,
    build_denoiser,
)
from ..core.global_vars import (
    get_combined_iterator,
    get_mm_state,
    get_denoiser,
    get_vae,
)
from ..data_kits.text_image_diffusion_loader import TextImageArrowStream
from ..models.autoencoders import vae_encode_and_add_noise
from ..models.text_encoder import TextEncoder


class Text2ImageDiffusionTrainer(MultimodalTrainer):

    def build_extra_model(self):
        """ initialize scalar states, denoiser, vae, tkwrapper as global vars. """
        args = self.args

        # ss maybe loaded from checkpoint, so only build when not exist.
        if not hasattr(self, "ss") or self.ss is None:
            self.ss = build_scalar_state()

        self.denoiser = build_denoiser()
        self.vae = build_vae()

        # ====================== Build text encoder ========================
        self.text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder_type,
            max_length=args.text_encoder_text_len,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer_name,
            use_attention_mask=args.text_encoder_use_attention_mask,
            infer_mode=args.text_encoder_infer_mode,
            prompt_template=(
                LI_DIT_PROMPT_TEMPLATE[args.text_encoder_prompt_template]
                if args.text_encoder_prompt_template is not None else None
            ),
            hidden_state_skip_layer=args.text_encoder_hidden_state_skip_layer,
            apply_final_norm=args.text_encoder_apply_final_norm,
            reproduce=args.reproduce,
            logger=self.logger,
            device=self.device,
            attn_implementation=getattr(args, "text_encoder_attn", None),
        )

    def build_dataloader(self):
        DatasetsProvider(
            task_info_dict={
                "t2i*": dict(cur_task="t2i", cls=TextImageArrowStream),
            }
        )(None)
        self.combined_iterator = get_combined_iterator()
        self.mm_state = get_mm_state()

    def prepare_model_inputs(self, batch, device, **kwargs):
        # Move batch to device
        image_tensor = batch['image_tensor'].to(device)
        freqs_cos = batch['freqs_cos'].to(device)
        freqs_sin = batch['freqs_sin'].to(device)

        bsz = image_tensor.shape[0]
        n_tokens = freqs_cos.shape[1]
        denoiser = get_denoiser()

        # ======================================== Encode media ======================================
        if kwargs.get('skip_vae_encode'):
            t, model_t, x_0, x_t, u_t = None, None, None, None, None
        else:
            vae = get_vae()
            out = vae_encode_and_add_noise(vae, image_tensor, device, denoiser=denoiser, sample_type="sample")
            t, model_t, x_0, x_t, u_t = out.t, out.model_t, out.x_0, out.x_t, out.u_t
        
        diffusion_loss_fn = partial(denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t)

        # ======================================== Encode text ======================================
        # Autocast is handled by text_encoder itself.
        # Whether to apply text_mask is determined by args.use_attention_mask.
        prompt = batch["prompt"]
        text_inputs = self.text_encoder.text2tokens(prompt)
        text_ids = text_inputs["input_ids"]
        text_mask = text_inputs["attention_mask"]
        text_ids = text_ids.to(device)
        text_mask = text_mask.to(device) if self.args.text_encoder_use_attention_mask else None

        text_outputs = self.text_encoder.encode({"input_ids": text_ids, "attention_mask": text_mask})
        text_states = text_outputs.hidden_state
        text_mask = text_outputs.attention_mask

        # ===================================== Pack model kwargs ==================================
        model_intput_kwargs = dict(x=x_t,
                                   t=model_t,
                                   cond=text_states,
                                   cond_mask=text_mask,
                                   freqs_cos=freqs_cos,            # [seqlen, head_dim]
                                   freqs_sin=freqs_sin,            # [seqlen, head_dim]
                                   return_dict=True,
                                   diffusion_loss_fn=diffusion_loss_fn)

        save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, bsz, n_tokens
