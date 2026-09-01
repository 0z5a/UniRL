import torch

from .pretrain_pure_torch import MultimodalTrainer
from ..core.data_provider_dit import prepare_model_inputs
from ..core.extra_model_provider import (
    build_text_encoder,
    build_audio_vae,
    build_denoiser,
    build_repa_encoder,
)


class Leo2Trainer(MultimodalTrainer):

    # vae 内部维护了 generator, 为了正确 resume，需要记录 vae generator 状态
    # 这里 tuple 含义为 [trainer 属性名, 保存到 client_state 的 key 名]
    # 会从这些属性中调用 generator.get_state() 和 generator.set_state() 方法
    _GENERATOR_STATE_BINDINGS = (
        ("vae", "vae_generator_state"),
        ("audio_vae", "audio_vae_generator_state"),
    )

    def build_extra_model(self):
        super().build_extra_model()
        args = self.args

        assert args.use_text_encoder, "`use_text_encoder` must be True for Leo2Trainer."
        self.text_encoder = build_text_encoder()

        # Modality-specific SNR config; fall back to global args when not set (None).
        video_snr_type = args.flow_snr_type_video or args.flow_snr_type
        video_snr_mix_ratio = (
            args.flow_snr_mix_uniform_ratio_video
            if args.flow_snr_mix_uniform_ratio_video is not None
            else args.flow_snr_mix_uniform_ratio
        )
        self.video_denoiser = build_denoiser(
            denoiser_type="video",
            shift=args.flow_shift_video,
            snr_type=video_snr_type,
            snr_mix_uniform_ratio=video_snr_mix_ratio,
        )
        if args.audio_branch_model_name is not None:
            audio_snr_type = args.flow_snr_type_audio or args.flow_snr_type
            audio_snr_mix_ratio = (
                args.flow_snr_mix_uniform_ratio_audio
                if args.flow_snr_mix_uniform_ratio_audio is not None
                else args.flow_snr_mix_uniform_ratio
            )
            self.audio_denoiser = build_denoiser(
                denoiser_type="audio",
                shift=args.flow_shift_audio,
                snr_type=audio_snr_type,
                snr_mix_uniform_ratio=audio_snr_mix_ratio,
            )

        self.audio_vae = build_audio_vae(use_audio_vae=args.use_audio_vae)

        if args.use_repa:
            self.repa_encoder = build_repa_encoder()

    
    def _collect_extra_client_state(self) -> dict:
        state = super()._collect_extra_client_state()
        for attr_name, key in self._GENERATOR_STATE_BINDINGS:
            obj = getattr(self, attr_name, None)
            if obj is not None and getattr(obj, "generator", None) is not None:
                state[key] = obj.generator.get_state().clone().cpu()
        return state

    def _consume_extra_client_state(self, client_state: dict) -> None:
        super()._consume_extra_client_state(client_state)
        for attr_name, key in self._GENERATOR_STATE_BINDINGS:
            state = client_state.get(key)
            if state is None:
                continue
            obj = getattr(self, attr_name, None)
            if obj is not None and getattr(obj, "generator", None) is not None:
                obj.generator.set_state(state)
                self.logger.info(f"Restored {attr_name}.generator state from checkpoint.")

    def after_initialize(self):
        super().after_initialize()

        args = self.args

        if args.audio_branch_model_name is not None:
            self.loss_names.extend([
                f"{dataset_tag}_audio_loss"
                for dataset_tag in self.mm_state.all_dataset_keys
                if "2a" in dataset_tag or "2va" in dataset_tag
            ])

        if args.moe_aux_loss and args.moe_aux_loss_coeff > 0:
            config = self.model.get_config()
            txt_config = self.model.get_txt_config()

            if config.num_experts > 0:
                self.loss_names.extend(["image_video_moe_loss", "image_video_capacity_rate"])
            if txt_config.num_experts > 0:
                self.loss_names.extend(["text_moe_loss", "text_capacity_rate"])

        if self.args.use_repa:
            self.loss_names.append("repa_loss")

        if getattr(args, "use_global_diffusion_loss_average", False):
            self.loss_names.extend(["global_image_video_loss", "global_audio_loss"])
            if self.args.use_repa:
                self.loss_names.append("global_repa_loss")

    def prepare_model_inputs(self, batch, device):
        with torch.autocast(device_type="cuda", enabled=False):
            model_input_kwargs, bsz, seqlen = prepare_model_inputs(
                batch, device, model_config=self.model.get_config(),
            )
        return model_input_kwargs, bsz, seqlen

    @torch.no_grad()
    def validation(self, timer):
        from hymm.samplers.leo2_sampler import Leo2Sampler

        timer.start("validation", sync=True, barrier=True)
        sampler = Leo2Sampler(None, self.rank, self.world_size, trainer=self)
        sampler.run_testsets(
            sample_save_base=self.exp_dir / f"samples/iter_{self.ss.update_steps:07d}",
        )
        timer.stop("validation")

    def apply_global_loss_average(self, loss_dict, apply_global_loss_average_fn, batch=None):
        args = self.args

        # Optionally average diffusion loss over global image sample count (all ranks)
        if getattr(args, "use_global_diffusion_loss_average", False):
            apply_global_loss_average_fn(
                sum_key="diff_loss_sum",
                count_key="diff_loss_count",
                loss_weight=loss_dict.pop("diff_loss_weight", args.image_loss_weight),
                global_loss_name="global_image_video_loss",
                loss_dict=loss_dict,
                gbca_count=None if batch is None else batch.get("micro_batch_image_count"),
            )
            apply_global_loss_average_fn(
                sum_key="audio_diff_loss_sum",
                count_key="audio_diff_loss_count",
                loss_weight=loss_dict.pop("audio_diff_loss_weight", args.audio_loss_weight),
                global_loss_name="global_audio_loss",
                loss_dict=loss_dict,
                gbca_count=None if batch is None else batch.get("micro_batch_audio_count"),
            )
            if getattr(args, "use_repa", False):
                apply_global_loss_average_fn(
                    sum_key="repa_loss_sum",
                    count_key="repa_loss_count",
                    loss_weight=loss_dict.pop("repa_loss_weight", args.repa_loss_weight),
                    global_loss_name="global_repa_loss",
                    loss_dict=loss_dict,
                    gbca_count=None if batch is None else batch.get("micro_batch_repa_count"),
                )