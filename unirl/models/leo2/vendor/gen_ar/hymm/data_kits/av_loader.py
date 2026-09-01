import json
import random
from itertools import groupby

import torch
from loguru import logger as _logger

from .mm_loader import MultimodalIndexDataset
from .utils import (
    AudioMixin,
    AudioCaptionMixin,
    VideoMixin,
    VideoCaptionMixin,
    resample_for_errors,
)
from .utils.data_container import MultimodalDataContainer, maybe_stack
from .utils.video_utils import merge_qwen3vl_video_grid_thw
from ..utils.image_base import CondImage
from ..constants import SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES
from ..core.global_vars import get_parallel_state


class MultimodalAVIndexDataset(
    MultimodalIndexDataset, VideoMixin, VideoCaptionMixin, AudioMixin, AudioCaptionMixin
):
    def __post_init__(self, **kwargs):
        args = self.args

        # Leave some room for dummy tokens when building the sequence
        self.dummy_number_dict = dict(
            # image_ts_proj=2,  # 1 for patch_embed, and 1 for timestep
            # audio_ts_proj=2,  # 1 for patch_embed, and 1 for timestep
            audios=1 if args.audio_branch_model_name is not None else 0,
            visual=1 if "vae_video" in self.modality or "vae_image" in self.modality else 0,
        )

        self.audio_token_type = self.args.audio_token_type  # gen_audio token type

        # Attn type
        self.attn_type = self.task_kwargs.get('attn_type', 'auto')

        # Channel-concat extension type for the visual diffusion path.
        if args.extend_latent_channels:
            self.latent_channel_extend_type = self.task_kwargs.get("latent_channel_extend_type")
            assert self.latent_channel_extend_type in SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES, (
                f"`latent_channel_extend_type` must be set in `{self.dataset_tag}_task_kwargs` "
                f"to one of {list(SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES)} when "
                f"`extend_latent_channels=True`, got {self.latent_channel_extend_type!r}."
            )
        else:
            self.latent_channel_extend_type = None
            if self.task_kwargs.get("latent_channel_extend_type") is not None:
                self.logger.warning(
                    f"{self.log_prefix}`latent_channel_extend_type` is set in "
                    f"`{self.dataset_tag}_task_kwargs` but the global flag "
                    f"`extend_latent_channels` is False; the per-task setting is ignored."
                )

        # Sequence pack related and affected
        if self.sequence_pack:
            assert args.gen_template == "multi_stream_dit", \
                "Sequence pack is only supported for multi_stream_dit template for now."
            # Make seq_length - 1 to keep consistency with AR model because dit sequence doesn't need sequence shift.
            self.seq_length = self.seq_length - 1
            self.max_sequence_length = self.seq_length + 1 - sum(self.dummy_number_dict.values()) - self.reserved_pad_length
            self.logger.info(f"{self.log_prefix}Sequence pack is enabled with "
                             f"{self.seq_length=} and {self.max_sequence_length=}.")

        # ===========================================================================
        # Modality specific setup

        # VIDEO: setup vae info
        if "vae_video" in self.modality:
            self.setup_video(args)

        # AUDIO: setup vae info
        if "vae_audio" in self.modality:
            self.setup_audio(args)

        # ===========================================================================
        # TASK specific setup

        # VIDEO CAPTION: setup caption manager. For T2V, T2VA, I2V, FL2V, R2V
        if "t2v" in self.dataset_tag or "i2v" in self.dataset_tag or "fl2v" in self.dataset_tag \
                or "r2v" in self.dataset_tag:
            self.setup_video_caption(args)

        if "r2v" in self.dataset_tag:
            self.cond_after_gen = bool(self.task_kwargs.get("cond_after_gen", True))

        # AUDIO CAPTION: setup caption manager. For T2A, T2VA
        # When t2va using separated video caption and audio caption, we need setup_audio_caption,
        # otherwise the audio caption should already be included in the video caption, and we can
        # skip setup_audio_caption.
        # NOTICE: Separated caption will be deprecated in the future.
        self.audio_separated_caption = self.task_kwargs.get("audio_separated_caption", False)
        self.audio_caption_separator = self.task_kwargs.get("audio_caption_separator", "\nAudio: ")
        # For caption in t2a task, we need a template for aligning the caption template of t2va.
        self.audio_caption_template = self.task_kwargs.get("audio_caption_template", None)
        if "t2a" in self.dataset_tag or ("t2va" in self.dataset_tag and self.audio_separated_caption):
            self.setup_audio_caption(args)

    def get_rope_media_info(self, sections, output, data):
        rope_type = self.args.rope_type_extended or "leo2_3d"

        if rope_type == "leo2_3d":
            assert not self.sequence_pack, "Sequence pack is not supported for leo2_3d rope type"

            if data.images is not None:
                # t2i task
                assert len(data.images) == 1, f"Only support one image for now, but got {len(data.images)} images."
                rope_media_info = [(None, (1, data.images[0].i.tk_h, data.images[0].i.tk_w), {"type": "gen_image"})]
            elif data.audios is not None and data.videos is not None:
                # t2va task
                assert len(data.audios) == 1, f"Only support one audio for now, but got {len(data.audios)} audios."
                assert len(data.videos) == 1, f"Only support one video for now, but got {len(data.videos)} videos."
                # Notice that video's tk_d is not equal to audio's tk_d.
                rope_media_info = [
                    (
                        None,
                        (data.videos[0].i.tk_d, data.videos[0].i.tk_h, data.videos[0].i.tk_w),
                        {"type": "gen_video"}
                    ),
                    (None, (data.audios[0].i.tk_d,), {"type": "gen_audio"})
                ]
            elif data.audios is not None:
                # t2a task
                assert len(data.audios) == 1, f"Only support one audio for now, but got {len(data.audios)} audios."
                rope_media_info = [(None, (data.audios[0].i.tk_d,), {"type": "gen_audio"})]
            elif data.videos is not None:
                # t2v, i2v, fl2v tasks without audio
                assert len(data.videos) == 1, f"Only support one video for now, but got {len(data.videos)} videos."
                rope_media_info = [(
                    None,
                    (data.videos[0].i.tk_d, data.videos[0].i.tk_h, data.videos[0].i.tk_w),
                    {"type": "gen_video"}
                )]
            else:
                raise ValueError(f"No media found in data for rope media info processing. data: {data}")
            num_overlapped = 0

        elif rope_type == "3d":
            raw_media_slices = output.all_media_slices
            media_idx = 0
            media_slices = []
            media_shapes = []
            metas = []
            for section in sections:
                if media_idx > len(raw_media_slices):
                    break
                if section["type"] in ["gen_image", "gen_video", "gen_audio"]:
                    media_slices.append(raw_media_slices[media_idx])
                    metas.append({'type': section["type"]})
                    if section["type"] == "gen_image":
                        media_shapes.append((1, section["token_height"], section["token_width"]))
                    elif section["type"] == "gen_video":
                        media_shapes.append((section["token_duration"], section["token_height"], section["token_width"]))
                        metas[-1]["with_audio"] = section.get("with_audio", False)
                        metas[-1]["dataset_tag"] = self.dataset_tag
                    elif section["type"] == "gen_audio":
                        media_shapes.append((section["token_length"], 1, 1))
                        metas[-1]["with_video"] = section.get("with_video", False)
                        if not section.get("with_video", False):
                            metas[-1]["rope_audio_rescale_factor"] = getattr(self.args, "rope_audio_rescale_factor", 1.0)
                    media_idx += 1

                elif section["type"] in ["cond_vae_image", "cond_vit_image"]:
                    media_slices.append(raw_media_slices[media_idx])
                    media_shapes.append((1, section["token_height"], section["token_width"]))
                    metas.append({'type': section["type"]})
                    media_idx += 1

                elif section["type"] in ["cond_vae_video", "cond_vit_video"]:
                    media_slices.append(raw_media_slices[media_idx])
                    media_shapes.append((section["token_duration"], section["token_height"], section["token_width"]))
                    meta = {'type': section["type"]}
                    # Tubelets of a native condition video share a RoPE base and advance along time.
                    for key in ("temporal_index", "temporal_length", "video_id", "timestamp"):
                        if key in section:
                            meta[key] = section[key]
                    metas.append(meta)
                    media_idx += 1

                elif section["type"] == "text":
                    pass

                else:
                    raise ValueError(f"Unsupported section type: {section['type']} in rope media info processing.")

            rope_media_info = list(map(list, zip(media_slices, media_shapes, metas)))
            num_overlapped = 0

        else:
            raise ValueError(f"Unsupported rope_type_extended: {rope_type}")

        return rope_media_info, num_overlapped

    @resample_for_errors
    def get_t2v_data(self, index) -> MultimodalDataContainer:
        # For t2v, i2v, fl2v, t2va tasks.
        self.require_configs(self.index_columns, "video_col",
                             f"{self.dataset_tag}.{self.dataset_tag}_index_kwargs.index_columns")
        is_fl2v = self.latent_channel_extend_type == "fl2v"
        lf_is_latent = getattr(self, "last_frame_data_format", "pixels") == "latents" # last_frame_data_format: 是否离线提取latent，值为latents是离线提取，pixels是在线提取。离线提取的latent不带normalization，需要做normalize
        if is_fl2v:
            required_col = "last_frame_latent_col" if lf_is_latent else "last_frame_col"
            self.require_configs(self.index_columns, required_col,
                                 f"{self.dataset_tag}.{self.dataset_tag}_index_kwargs.index_columns")

        # Drop blacklisted samples (see `index_filter_kwargs`) before touching COS, so a filtered sample
        # costs no video/audio latent nor caption read. `success=False` lets the `resample_for_errors`
        # decorator substitute a random other sample, keeping the step count unchanged. If every retry is
        # blacklisted too, the empty container reaches the tokenizer, whose `prompt=None` raises the
        # TypeError that `__getitem__` already recovers from.
        if self.hit_index_filter(index):
            return MultimodalDataContainer(
                success=False,
                index=index,
                dataset_tag=self.dataset_tag,
                error="filtered data",
            )

        # Get video
        tgt_video, video_success = self.get_video_with_size(
            index,
            return_type="vae",
            **self.index_columns["video_col"],
        )

        # Get conditional last frame (pixels or offline latent, decided by global format)
        if is_fl2v:
            lf_col = self.index_columns["last_frame_latent_col"] if lf_is_latent \
                else self.index_columns["last_frame_col"]
            last_frame_tensor, last_frame_success = self.get_last_frame_with_size(
                src=index,
                latent_size=tgt_video.size(),
                **lf_col,
            )
            # Increase token_duration of the tgt_video to make room for
            # the appended last frame in the channel-concat path.
            tgt_video.i.inc_token_duration()
        else:
            last_frame_tensor, last_frame_success = None, True

        # Get audio
        if "vae_audio" in self.task_kwargs["modality"]:
            audio_data_format = self.task_kwargs.get("audio_data_format", "single_slice_file")
            tgt_audio, audio_success = self.get_audio_with_size(
                index,
                return_type="vae",
                data_format=audio_data_format,
                **self.index_columns["audio_col"],
            )
        else:
            tgt_audio, audio_success = None, True

        # Get conditional video/av text
        cap_out, cap_success = self.get_video_caption(index, tgt_video)
        # Get conditional separated audio text
        if self.task_kwargs.get("audio_separated_caption", False):
            audio_cap_out, audio_cap_success = self.get_audio_caption(index)
            cap_success = cap_success and audio_cap_success
            if audio_cap_success:
                cap_out.caption += self.audio_caption_separator + audio_cap_out.caption

        # -- user prompt
        prompt = cap_out.caption

        return MultimodalDataContainer(
            prompt=prompt,
            videos=[tgt_video],
            video_last_frames=[last_frame_tensor] if last_frame_tensor is not None else None,
            audios=[tgt_audio] if tgt_audio is not None else None,
            success=video_success and audio_success and cap_success and last_frame_success,
            index=index,
            dataset_tag=self.dataset_tag,
        )

    def build_t2v_template(self, data: MultimodalDataContainer) -> list[dict]:
        # Unconditional
        do_uncond = (self.video_uncond_p > 0) and (random.random() < self.video_uncond_p)
        uncond_kwargs = dict(
            uncond_enabled=do_uncond,
            uncond_p=(1.0 if do_uncond else 0.0),
            uncond_length=self.args.uncond_length,
        )

        # Sequence decorator sections
        deco = self.decorator_sections
        # System prompt section
        if self.system_prompt_token_length > 0:
            system_prompt_section = self.get_system_prompt(data, return_section=True)
        else:
            system_prompt_section = []

        # prompt/generate sections
        assert self.video_prompt_token_length > 0, \
            f"self.video_prompt_token_length should be greater than 0, got {self.video_prompt_token_length}"
        prompt_section = [
            dict(type="text", text=data.prompt, max_length=self.video_prompt_token_length,
                 **uncond_kwargs),
        ]
        # NOTICE: here we set `ignore=False` for system prompt and prompt sections to get a text mask for
        #         the text encoder later.
        gen_section = []
        if data.videos is not None:
            gen_section.append(dict(type="gen_video", **data.videos[0].i.meta_info, with_audio=data.audios is not None))
        if data.audios is not None:
            gen_section.append(dict(type="gen_audio", **data.audios[0].i.meta_info, with_video=True))

        # Compose all sections
        if self.args.gen_template == "dit_it":
            sections = (
                    system_prompt_section +
                    deco.user + prompt_section + deco.user_sep
            )
        else:
            sections = (
                    system_prompt_section +
                    deco.user + prompt_section + deco.user_sep +
                    deco.bot + deco.answer(gen_section) + deco.bot_sep
            )

        return sections

    # =========================================================================
    # R2V (reference/source-video conditioned generation) helpers
    # =========================================================================

    @staticmethod
    def _sort_r2v_messages(raw_messages, index):
        if isinstance(raw_messages, str):
            try:
                raw_messages = json.loads(raw_messages)
            except json.JSONDecodeError as e:
                raise ValueError(f"r2v message column is a string but not valid JSON ({index=}): {e}") from e
        if not isinstance(raw_messages, (list, tuple)) or not raw_messages:
            raise ValueError(f"r2v message must be a non-empty list, got {type(raw_messages)} ({index=}).")
        if not all(isinstance(msg, dict) for msg in raw_messages):
            raise TypeError(f"Every r2v message item must be a dict ({index=}).")
        if any(msg.get("index") is None for msg in raw_messages):
            raise ValueError(f"Every r2v message item must contain a non-null `index` ({index=}).")

        sorted_messages = sorted(raw_messages, key=lambda msg: int(msg["index"]))
        message_indices = [int(msg["index"]) for msg in sorted_messages]
        if len(message_indices) != len(set(message_indices)):
            raise ValueError(f"Duplicate message indices {message_indices} ({index=}).")
        if message_indices != list(range(len(message_indices))):
            raise ValueError(
                f"Message indices must be contiguous and start at 0, got {message_indices} ({index=})."
            )
        return sorted_messages

    @resample_for_errors
    def get_r2v_data(self, index) -> MultimodalDataContainer:
        """Fetch one r2v sample from the typed interleave `message` column."""
        self.require_configs(
            self.index_columns,
            "message_col",
            f"{self.dataset_tag}.{self.dataset_tag}_index_kwargs.index_columns",
        )
        self.require_configs(
            self.index_keys,
            ["image_key", "video_key", "audio_key"],
            f"{self.dataset_tag}.{self.dataset_tag}_index_kwargs.index_keys",
        )

        raw_messages = self.index_manager.get_attribute(
            index, **self.index_columns["message_col"]
        )
        raw_messages = self._sort_r2v_messages(raw_messages, index)
        audio_data_format = self.task_kwargs.get("audio_data_format", "single_slice_latent")
        # ViT conditioning reads the source video's pixels
        video_frame_col = self.index_keys.get("video_material_key") if "vit" in self.cond_video_type else None

        messages = []
        gen_videos, gen_audios, cond_images, cond_videos, cond_audios = [], [], [], [], []
        all_media_success, all_text_success = True, True

        for msg in raw_messages:
            mtype = msg.get("type")
            if not mtype:
                raise ValueError(f"Missing message type ({index=}, message={msg}).")

            if mtype == "cond_caption":
                cap_out, cap_success = self.get_video_caption(msg)
                all_text_success = all_text_success and cap_success
                messages.append(dict(type="cond_text", text=cap_out.caption))

            elif mtype == "cond_image":
                cond_image, ok = self.get_image_with_size(
                    src=msg,
                    random_crop=False,
                    target_size_type="image",
                    return_type=self.cond_image_type,
                    real_index=index,
                    apply_exif=self.task_kwargs.get("apply_exif", False),
                    column=self.index_keys["image_key"],
                )
                cond_images.append(cond_image)
                messages.append(dict(type="cond_image", cond_image=cond_image))
                all_media_success = all_media_success and ok

            elif mtype == "cond_av":
                cond_video, vid_success = self.get_video_with_size(
                    src=msg,
                    return_type=self.cond_video_type,
                    real_index=index,
                    video_frame_col=video_frame_col,
                    video_id=len(cond_videos), # index依次递增
                    column=self.index_keys["video_key"],
                )
                cond_videos.append(cond_video)
                messages.append(dict(type="cond_video", cond_video=cond_video, with_audio=True))
                all_media_success = all_media_success and vid_success

                cond_audio, aud_success = self.get_audio_with_size(
                    src=msg,
                    return_type=self.cond_audio_type,
                    real_index=index,
                    data_format=audio_data_format,
                    column=self.index_keys["audio_key"],
                )
                cond_audios.append(cond_audio)
                messages.append(dict(type="cond_audio", cond_audio=cond_audio, with_video=True))
                all_media_success = all_media_success and aud_success

            elif mtype == "cond_video":
                cond_video, vid_success = self.get_video_with_size(
                    src=msg,
                    return_type=self.cond_video_type,
                    real_index=index,
                    video_frame_col=video_frame_col,
                    video_id=len(cond_videos),
                    column=self.index_keys["video_key"],
                )
                cond_videos.append(cond_video)
                messages.append(dict(type="cond_video", cond_video=cond_video))
                all_media_success = all_media_success and vid_success

            elif mtype == "gen_av":
                gen_video, vid_success = self.get_video_with_size(
                    src=msg,
                    return_type="vae",
                    real_index=index,
                    column=self.index_keys["video_key"],
                )
                gen_videos.append(gen_video)
                messages.append(dict(type="gen_video", gen_video=gen_video, with_audio=True))
                all_media_success = all_media_success and vid_success

                gen_audio, aud_success = self.get_audio_with_size(
                    src=msg,
                    return_type="vae",
                    real_index=index,
                    data_format=audio_data_format,
                    column=self.index_keys["audio_key"],
                )
                gen_audios.append(gen_audio)
                messages.append(dict(type="gen_audio", gen_audio=gen_audio, with_video=True))
                all_media_success = all_media_success and aud_success

            elif mtype == "gen_video":
                gen_video, vid_success = self.get_video_with_size(
                    src=msg,
                    return_type="vae",
                    real_index=index,
                    column=self.index_keys["video_key"],
                )
                gen_videos.append(gen_video)
                messages.append(dict(type="gen_video", gen_video=gen_video))
                all_media_success = all_media_success and vid_success

            elif mtype == "gen_audio":
                gen_audio, aud_success = self.get_audio_with_size(
                    src=msg,
                    return_type="vae",
                    real_index=index,
                    data_format=audio_data_format,
                    column=self.index_keys["audio_key"],
                )
                gen_audios.append(gen_audio)
                messages.append(dict(type="gen_audio", gen_audio=gen_audio))
                all_media_success = all_media_success and aud_success

            else:
                raise ValueError(f"Unsupported message type: {mtype}")

        if not gen_videos and not gen_audios:
            raise ValueError(f"r2v message must contain at least one generation target ({index=}).")

        return MultimodalDataContainer(
            cond_images=cond_images,
            cond_videos=cond_videos,
            cond_audios=cond_audios,
            videos=gen_videos,
            audios=gen_audios,
            messages=messages,
            success=all_media_success and all_text_success,
            index=index,
            dataset_tag=self.dataset_tag,
        )

    def build_r2v_template(self, data: MultimodalDataContainer) -> tuple[list[dict], "torch.Tensor | None"]:
        """Build tokenizer sections from an interleave `message` container.

        Successive messages of the same role are grouped into a single decorated block:
            system + user + [cond_*_image / text]* + user_sep
                  + bot + answer([gen_video / gen_audio]*) + bot_sep

        When `cond_after_gen` is set, the image/video conditioning is relocated to after the
        generation block (text prompt stays in front), yielding:
            system + user + [cond_text]* + user_sep
                  + bot + answer([gen_video / gen_audio]*) + bot_sep
                  + user + [cond_vit_image / cond_vae_image / cond_vit_video / cond_vae_video]* + user_sep

        For `dit_it` (inference) only the user-side blocks are emitted.
        """
        assert data.messages, "r2v message template requires non-empty messages."
        assert self.video_prompt_token_length > 0, (
            f"video_prompt_token_length should be > 0, got {self.video_prompt_token_length}"
        )

        deco = self.decorator_sections
        do_uncond = (self.video_uncond_p > 0) and (random.random() < self.video_uncond_p)
        uncond_kwargs = {
            "uncond_enabled": do_uncond,
            "uncond_p": 1.0 if do_uncond else 0.0,
            "uncond_length": self.args.uncond_length,
        }
        video_grids = []

        def vit_video_sections(vit_video, video_id):
            """Expand one native condition video into a section per temporal tubelet.

            Each tubelet carries its own timestamp so the language model can read the clip's
            temporal ordering, which a single fused block would hide.
            """
            grid_thw = merge_qwen3vl_video_grid_thw(vit_video)
            video_grids.append(grid_thw)

            info = vit_video.i
            num_tubelets = int(grid_thw[0, 0])
            timestamps = vit_video.vision_encoder_kwargs["timestamps"]
            tubelet_token_length = int(info.token_height) * int(info.token_width)
            if int(info.video_token_length) != num_tubelets * tubelet_token_length:
                raise ValueError(
                    "Condition-video token length is inconsistent with its temporal grid: "
                    f"{info.video_token_length} != {num_tubelets} * {tubelet_token_length}."
                )
            return [{
                "type": "cond_vit_video",
                "token_length": tubelet_token_length,
                "token_height": int(info.token_height),
                "token_width": int(info.token_width),
                "token_duration": 1,
                "timestamp": float(timestamp),
                "temporal_index": temporal_index,
                "temporal_length": num_tubelets,
                "video_id": int(video_id),
            } for temporal_index, timestamp in enumerate(timestamps)]

        def message_to_sections(message):
            mtype = message["type"]

            if mtype == "cond_text":
                return [dict(
                    type="text",
                    text=message["text"],
                    max_length=self.video_prompt_token_length,
                    **uncond_kwargs,
                )]

            elif mtype == "cond_image":
                cond_image = message["cond_image"]
                sections = []
                if getattr(cond_image, "vit_image", None) is not None:
                    sections.append(dict(type="cond_vit_image", **cond_image.vit_image.i.meta_info))
                if getattr(cond_image, "vae_image", None) is not None:
                    sections.append(dict(type="cond_vae_image", **cond_image.vae_image.i.meta_info))
                return sections

            elif mtype == "cond_video":
                cond_video = message["cond_video"]
                sections = []
                if getattr(cond_video, "vit_video", None) is not None:
                    sections.extend(vit_video_sections(
                        cond_video.vit_video, cond_video.vit_video.i.video_id or 0,
                    ))
                if getattr(cond_video, "vae_video", None) is not None:
                    sections.append(dict(type="cond_vae_video", **cond_video.vae_video.i.meta_info,
                                         with_audio=message.get("with_audio", False)))
                return sections

            elif mtype == "cond_audio":
                cond_audio = message["cond_audio"]
                return [
                    dict(type="cond_audio", **cond_audio.i.meta_info, with_video=message.get("with_video", False))
                ]

            elif mtype == "gen_video":
                gen_video = message["gen_video"]
                return [
                    dict(type="gen_video", **gen_video.i.meta_info, with_audio=message.get("with_audio", False))
                ]

            elif mtype == "gen_audio":
                gen_audio = message["gen_audio"]
                return [
                    dict(type="gen_audio", **gen_audio.i.meta_info, with_video=message.get("with_video", False))
                ]

            else:
                raise ValueError(f"Unsupported message type: {mtype}")

        def role_of(message):
            mtype = message["type"]
            if mtype.startswith("cond_"):
                return "cond"
            if mtype.startswith("gen_"):
                return "gen"
            raise ValueError(f"Unsupported internal r2v message role for type {mtype!r}.")

        # Optionally relocate the reference conditioning (`cond_image` / `cond_video` /
        # `cond_audio`) to *after* the generation target.
        messages = data.messages
        if self.cond_after_gen:
            cond_ref_types = ("cond_image", "cond_video", "cond_audio")
            cond_text_msgs = [m for m in messages
                              if role_of(m) == "cond" and m["type"] not in cond_ref_types]
            gen_msgs = [m for m in messages if role_of(m) == "gen"]
            cond_ref_msgs = [m for m in messages if m["type"] in cond_ref_types]
            messages = cond_text_msgs + gen_msgs + cond_ref_msgs

        assert self.args.gen_template == "multi_stream_dit", \
            f"gen_template should be 'multi_stream_dit', got {self.args.gen_template}"

        sections = []
        for role, message_group in groupby(messages, key=role_of):
            group_sections = []
            for message in message_group:
                group_sections.extend(message_to_sections(message))
            if role == "cond":
                sections.extend(deco.user + group_sections + deco.user_sep)
            elif role == "gen":
                sections.extend(deco.bot + deco.answer(group_sections) + deco.bot_sep)
        video_grid_thw = torch.cat(video_grids, dim=0) if video_grids else None
        return sections, video_grid_thw

    @resample_for_errors
    def get_t2a_data(self, index) -> MultimodalDataContainer:
        # For t2a task: audio only, no video.
        self.require_configs(self.index_columns, "audio_col",
                             f"{self.dataset_tag}.{self.dataset_tag}_index_kwargs.index_columns")

        audio_data_format = self.task_kwargs.get("audio_data_format", "single_slice_file")

        # Get audio caption
        cap_out, cap_success = self.get_audio_caption(index)
        caption_sample = {}
        start, end = None, None
        if audio_data_format.startswith("single_slice"):
            if cap_success and self.audio_caption_template is not None:
                cap_out.caption = self.audio_caption_template.format(caption=cap_out.caption)
        elif audio_data_format == "multi_slice_file":
            if cap_success:
                try:
                    caption_list = json.loads(cap_out.caption)
                    assert len(caption_list) > 0, f"len(caption_list): {len(caption_list)} should be greater than 0."

                    valid_caption_list = []
                    for caption_list_i in caption_list:
                        if caption_list_i["end"] - caption_list_i["start"] < 60:
                            valid_caption_list.append(caption_list_i)
                    assert len(valid_caption_list) > 0, (
                        f"No caption_sample with (end - start) < 60s found in "
                        f"caption_list of size {len(caption_list)}."
                    )
                    caption_sample = random.choice(valid_caption_list)

                    start = caption_sample["start"]
                    end = caption_sample["end"]

                    if self.audio_caption_template is not None:
                        cap_out.caption = self.audio_caption_template.format(
                            caption=json.loads(caption_sample["caption"])["long_detailed_description"])
                    else:
                        cap_out.caption = json.loads(caption_sample["caption"])["long_detailed_description"]
                except Exception as e:
                    # print(f"Error in json.loads(caption_sample['caption'])['long_detailed_description']: {e}, "
                    #       f"caption_sample['caption']={caption_sample["caption"]}")
                    cap_success = False
        else:
            raise ValueError(f"audio_data_format {audio_data_format} not supported")

        # Get audio
        tgt_audio, audio_success = self.get_audio_with_size(
            index,
            return_type="vae",
            data_format=audio_data_format,
            start=start,
            end=end,
            **self.index_columns["audio_col"],
        )

        # -- user prompt
        prompt = cap_out.caption

        return MultimodalDataContainer(
            prompt=prompt,
            audios=[tgt_audio],
            success=audio_success and cap_success,
            index=index,
            dataset_tag=self.dataset_tag,
        )

    def build_t2a_template(self, data: MultimodalDataContainer) -> list[dict]:
        # Unconditional
        do_uncond = (self.audio_uncond_p > 0) and (random.random() < self.audio_uncond_p)
        uncond_kwargs = dict(
            uncond_enabled=do_uncond,
            uncond_p=(1.0 if do_uncond else 0.0),
            uncond_length=self.args.uncond_length,
        )

        # Sequence decorator sections
        deco = self.decorator_sections
        # System prompt section
        if self.system_prompt_token_length > 0:
            system_prompt_section = self.get_system_prompt(data, return_section=True)
        else:
            system_prompt_section = []

        # prompt/generate sections
        assert self.audio_prompt_token_length > 0, \
            f"self.audio_prompt_token_length should be greater than 0, got {self.audio_prompt_token_length}"
        prompt_section = [
            dict(type="text", text=data.prompt, max_length=self.audio_prompt_token_length,
                 **uncond_kwargs),
        ]
        gen_section = [
            dict(type="gen_audio", **data.audios[0].i.meta_info),
        ]
        # Compose all sections
        if self.args.gen_template == "dit_it":
            sections = (
                    system_prompt_section +
                    deco.user + prompt_section + deco.user_sep
            )
        else:
            sections = (
                    system_prompt_section +
                    deco.user + prompt_section + deco.user_sep +
                    deco.bot + deco.answer(gen_section) + deco.bot_sep
            )

        return sections

    def __getitem__(self, index):
        index = int(index)
        video_grid_thw = None

        # Get data and build template
        if self.dataset_tag.startswith("t2i"):
            data = self.get_t2i_data(index)
            sections = self.build_t2i_template(data, num_predicted_image_token_offsets=(0, 0))

        elif self.dataset_tag.startswith("r2v"):
            data = self.get_r2v_data(index)
            sections, video_grid_thw = self.build_r2v_template(data)

        elif self.dataset_tag.startswith("t2v") or self.dataset_tag.startswith("i2v") \
                or self.dataset_tag.startswith("fl2v") or self.dataset_tag.startswith("t2va"):
            data = self.get_t2v_data(index)
            sections = self.build_t2v_template(data)

        elif self.dataset_tag.startswith("t2a"):
            data = self.get_t2a_data(index)
            sections = self.build_t2a_template(data)

        else:
            raise ValueError(f"Unsupported dataset tag: {self.dataset_tag}")

        dummy_type_dict = {}
        if self.args.audio_branch_model_name is not None and data.audios is None:
            dummy_type_dict["audio"] = 1    # audio branch dummy token
        if data.videos is None and data.images is None:
            dummy_type_dict["visual"] = 1   # visual branch dummy token
        dummy_number = sum(dummy_type_dict.values())

        max_token_length = self.seq_length - dummy_number \
            if self.sequence_pack \
            else self.max_token_length - dummy_number

        try:
            output = self.tokenizer.encode_general(
                sections=sections,
                max_token_length=max_token_length,
                add_eos=False,
                drop_last=self.drop_last,
                add_pad=False if self.sequence_pack else 'auto',
                add_bos=self.default_conv.add_bos if hasattr(self, 'default_conv') else True,
                und_token_type=self.und_token_type,
                gen_token_type=self.gen_token_type,
                audio_token_type=self.audio_token_type,
                disable_ignore=True,    # Get the text_mask including all text tokens. Text `ignore` only used for AR.
            )
        except TypeError as e:
            self.logger.error(
                f"TypeError in encoding sections (dataset_tag={self.dataset_tag}, index={data.index}): "
                f"{self.max_token_length=}, {self.sequence_pack=}, {sections=}. Original error: {e}"
            )
            return self[(index + 100000) % len(self)]
        except AssertionError as e:
            self.logger.error(
                f"Error in encoding sections (dataset_tag={self.dataset_tag}, index={data.index}): "
                f"{self.max_token_length=}, {self.sequence_pack=}, {sections=}"
            )
            raise e

        # text_mask is [0, 0, ..., 1, 1, 1, ..., 0, 0], where 1s for text tokens and tailing 0s for pad tokens.
        # The leading 0s are for conditional media tokens.
        output.text_mask = output.text_mask.to(torch.long)

        # Remove empty slices
        text_slices = [sli for sli in output.text_slices if sli.stop > sli.start]

        # Attention mask will be created in data_provider_dit.
        assert self.attn_type != 'auto', 'Notimplemented.'
        attention_mask = None

        rope_media_info, num_overlapped = self.get_rope_media_info(sections, output, data)

        # Compose vit-image kwargs (for Qwen3-VL multimodal text encoder) when present.
        cond_vit_image_kwargs = None
        if data.cond_vit_images and getattr(data.cond_vit_images[0], "vision_encoder_kwargs", None) is not None:
            image_type = getattr(data.cond_vit_images[0].i, "image_type", None)
            if image_type == "qwen3vl":
                cond_vit_image_kwargs = {
                    "grid_thw": maybe_stack(
                        data.cond_vit_images, key="vision_encoder_kwargs", subkey="grid_thw"
                    ),
                }
            else:
                cond_vit_image_kwargs = {
                    "spatial_shapes": maybe_stack(
                        data.cond_vit_images, key="vision_encoder_kwargs", subkey="spatial_shapes"
                    ),
                    "attention_mask": maybe_stack(
                        data.cond_vit_images, key="vision_encoder_kwargs", subkey="pixel_attention_mask"
                    ),
                }

        # Native condition videos travel on Qwen3-VL's dedicated video input, keyed by a
        # (num_videos, 3) grid whose temporal axis counts tubelets rather than frames.
        cond_vit_video_kwargs = None
        if video_grid_thw is not None:
            cond_vit_video_kwargs = {"video_grid_thw": video_grid_thw}

        ret = {
            # === required ===
            "dataset_tag": self.dataset_tag,
            "n_samples": 1,  # ()
            "index": data.index,  # ()
            "status": data.status,  # ()

            "tokens": output.tokens,  # (seqlen)
            "text_mask": output.text_mask,  # (seqlen)
            "text_slices": text_slices,
            # === optional ===
            "attention_mask": attention_mask,  # (1, seqlen - 1, seqlen - 1)
            # -- gen image
            "images": maybe_stack(data.images),
            "image_mask": output.gen_image_mask,
            "image_slices": output.gen_image_slices,
            # -- gen video
            "videos": maybe_stack(data.videos),
            "video_mask": output.gen_video_mask,
            "video_slices": output.gen_video_slices,
            # -- i2v or fl2v
            "video_last_frames": maybe_stack(data.video_last_frames),  # (n2, 3, H, W) or [n2, (3, H, W)]
            # -- r2v reference images (subject_img + background_img)
            "cond_vae_images": maybe_stack(data.cond_vae_images),       # (n_c, 3, H, W) or [n_c, ...]
            "cond_vae_image_mask": output.vae_image_mask,               # (seqlen)
            "cond_vae_image_slices": output.vae_image_slices,           # [n_c]
            "cond_vit_images": maybe_stack(data.cond_vit_images),       # (n_c, vit_seqlen, ndim) or [n_c, ...]
            "cond_vit_image_mask": output.vit_image_mask,               # (seqlen)
            "cond_vit_image_slices": output.vit_image_slices,           # [n_c]
            "cond_vit_image_kwargs": cond_vit_image_kwargs,             # spatial_shapes/grid_thw etc.
            # -- r2v source video: VAE latent (-> gen stream) and ViT frames (-> und stream)
            "cond_vae_videos": maybe_stack(data.cond_vae_videos),       # [n_v, (C, T, H, W)]
            "cond_vae_video_mask": output.vae_video_mask,               # (seqlen)
            "cond_vae_video_slices": output.vae_video_slices,           # [n_v]
            "cond_vit_videos": maybe_stack(data.cond_vit_videos),       # [n_v, (vit_seqlen, ndim)]
            "cond_vit_video_mask": output.vit_video_mask,               # (seqlen)
            "cond_vit_video_slices": output.vit_video_slices,           # [n_tubelets] dense ViT tokens only
            # context slices also cover each tubelet's timestamp text and delimiter tokens
            "cond_vit_video_context_slices": output.vit_video_context_slices,
            "cond_vit_video_kwargs": cond_vit_video_kwargs,             # video_grid_thw (n_videos, 3)
            # -- gen audio
            "audios": maybe_stack(data.audios),
            "audio_mask": output.gen_audio_mask,
            "audio_slices": output.gen_audio_slices,
            # -- position related
            "timesteps_index": output.gen_timestep_scatter_index,
            "video_timesteps_index": output.gen_video_timestep_scatter_index,
            "audio_timesteps_index": output.gen_audio_timestep_scatter_index,
            "rope_media_info": rope_media_info,
            # -- others
            "num_overlapped": num_overlapped,
            # -- dummy
            "dummy_type_dict": dummy_type_dict,
        }
        if self.args.use_mot or self.args.gen_template == "multi_stream_dit":
            # Only for packed mode
            ret["und_token_indices"] = output.und_token_indices
            ret["gen_token_indices"] = output.gen_token_indices
            ret["audio_token_indices"] = output.audio_token_indices

        ret = {k: v for k, v in ret.items() if v is not None}

        return ret

    def collate_fn(self, batch):
        if self.sequence_pack:
            return batch

        if self.drop_resampled_samples:
            filtered_batch = [item for item in batch if item["status"] == "original"]
            if len(filtered_batch) == 0:
                # If all samples in the batch are resampled, we keep the first sample to avoid empty batch.
                filtered_batch = [batch[0]]
            batch = filtered_batch

        # ==== optional fields ====
        # We denote () as tensor shape, [] as list, n0, n1 as number of images, cond images(vae/vit), respectively.
        # - mask will be: stacked tensor(bsz, seqlen)
        # - slices will be: list of lists of slices[bsz, n, slice]
        # - images will be: a 5-D tensor(bsz, n0, 3, H, W) or list of 4-D tensors[bsz, (n0, 3, H, W)]
        #   or list of lists of 3-D tensors[bsz, [n0, (3, H, W)]]
        # - vae images will be: a 5-D tensor(bsz, n1, 3, H, W) or list of 4-D tensors[bsz, (n1, 3, H, W)]
        #   or list of lists of 3-D tensors[bsz, [n1, (3, H, W)]]
        # - vit images will be: a 4-D tensor(bsz, n, vit_seqlen, ndim) or list of 3-D tensors[bsz, (n, vit_seqlen, ndim)]
        # - timesteps_index will be: a 2-D tensor(bsz, n0) or list of 1-D tensors [bsz, (n0)]
        # - cond_timesteps_index will be: a 2-D tensor(bsz, n1) or list of 1-D tensors [bsz, (n1)]
        ret = {
            # === required ===
            "dataset_tag": [item["dataset_tag"] for item in batch],
            "n_samples": torch.tensor([item["n_samples"] for item in batch]),         # (bsz),
            "index": [item["index"] for item in batch],                               # (bsz)
            "tokens": torch.stack([item["tokens"] for item in batch]),                # (bsz, seqlen)
            "text_mask": torch.stack([item["text_mask"] for item in batch]),          # (bsz, seqlen)
            "text_slices": [item["text_slices"] for item in batch],                   # (bsz, n, slice)
            # === optional ===
            "attention_mask": maybe_stack(batch, key="attention_mask", strict=True),
            # -- gen image
            "images": maybe_stack(batch, key="images"),
            # -- gen video
            "videos": maybe_stack(batch, key="videos"),
            # -- i2v or fl2v
            "video_last_frames": maybe_stack(batch, key="video_last_frames"),
            # -- gen audio
            "audios": maybe_stack(batch, key="audios"),
            # -- position related
            "rope_media_info": [item["rope_media_info"] for item in batch],
            # -- dummy
            "dummy_type_dict": [item["dummy_type_dict"] for item in batch],
        }
        ret = {key: value for key, value in ret.items() if value is not None}

        return ret

    def postprocess_packed_indices(self, new_item, max_length, items):
        # Count padding number for checking if needed.
        pad_count = dict(und=0, gen=0, audio=0)

        if "und_token_indices" in new_item and "gen_token_indices" in new_item and "audio_token_indices" in new_item:
            # When cp_size > 1: align und/gen for context parallel (last token in und, both divisible by cp_size after slice+dummy).
            # Pad reserve is done at sequence pack time so we always have enough pad to align.
            und_count = new_item["und_token_indices"].shape[1]
            gen_count = new_item["gen_token_indices"].shape[1]
            audio_count = new_item["audio_token_indices"].shape[1] if "audio_token_indices" in new_item else 0
            token_indices_length = und_count + gen_count + audio_count
            pad_length = max_length - token_indices_length
            p_state = get_parallel_state()
            cp_size = p_state.cp_size

            if self.args.pack_seq_reduce_pad and (p_state.backend != "megatron" or cp_size <= 1):
                if self.task_kwargs.get("attn_type") == "flex":
                    assert 0 <= pad_length < 128
                else:
                    assert pad_length == 0

            if pad_length > 0:
                device = new_item["und_token_indices"].device
                dtype = new_item["und_token_indices"].dtype
                padded_indices = torch.arange(pad_length, device=device)[None] + token_indices_length
                padded_indices = padded_indices.to(dtype=dtype)

                if cp_size > 1 and p_state.backend == 'megatron':
                    und_dummy = 0   # For now, all tasks share entirely the same text branch parameters.
                    gen_dummy = items[0]["dummy_type_dict"].get("visual", 0)
                    audio_dummy = items[0]["dummy_type_dict"].get("audio", 0)
                    assert all(gen_dummy == item["dummy_type_dict"].get("visual", 0) for item in items), \
                        (f"gen_dummy should be the same for all items in the batch, but got "
                         f"gen_dummy={gen_dummy} and {[item['dummy_type_dict'].get('visual', 0) for item in items]}")
                    assert all(audio_dummy == item["dummy_type_dict"].get("audio", 0) for item in items), \
                        (f"audio_dummy should be the same for all items in the batch, but got "
                         f"audio_dummy={audio_dummy} and {[item['dummy_type_dict'].get('audio', 0) for item in items]}")
                    gen_pad = (cp_size - (gen_count + gen_dummy) % cp_size) % cp_size
                    audio_pad = (cp_size - (audio_count + audio_dummy) % cp_size) % cp_size
                    und_pad = pad_length - gen_pad - audio_pad
                    assert und_pad >= 0, \
                        (f"und_pad must be non-negative, got und_pad={und_pad} with pad_length={pad_length}, "
                         f"gen_pad={gen_pad}, audio_pad={audio_pad}. Please check if the max_token_length is properly "
                         f"set to accommodate the tokens and dummy tokens after packing. {max_length=}, "
                         f"{und_count=}, {gen_count=}, {audio_count=}, {und_dummy=}, {gen_dummy=}, {audio_dummy=}, "
                         f"{token_indices_length=}")

                    if und_pad > 0:
                        und_from_pad = padded_indices[:, :und_pad]
                        new_item["und_token_indices"] = torch.cat(
                            [new_item["und_token_indices"], und_from_pad], dim=1
                        )
                        pad_count["und"] = und_pad
                    if gen_pad > 0:
                        gen_from_pad = padded_indices[:, und_pad:und_pad + gen_pad]
                        new_item["gen_token_indices"] = torch.cat(
                            [new_item["gen_token_indices"], gen_from_pad], dim=1
                        )
                        pad_count["gen"] = gen_pad
                    if audio_pad > 0:
                        audio_from_pad = padded_indices[:, und_pad + gen_pad:]
                        new_item["audio_token_indices"] = torch.cat(
                            [new_item["audio_token_indices"], audio_from_pad], dim=1
                        )
                        pad_count["audio"] = audio_pad
                    und_final = new_item["und_token_indices"].shape[1]
                    gen_final = new_item["gen_token_indices"].shape[1]
                    audio_final = new_item["audio_token_indices"].shape[1] if "audio_token_indices" in new_item else 0
                    assert (und_final + und_dummy) % cp_size == 0 and (gen_final + gen_dummy) % cp_size == 0 \
                           and (audio_final + audio_dummy) % cp_size == 0, (
                        f"After data_provider slice and add_dummy: (und_final+und_dummy) and (gen_final+gen_dummy) "
                        f"and (audio_final+audio_dummy) must be divisible by cp_size={cp_size}, but got "
                        f"und_final={und_final}, gen_final={gen_final}, audio_final={audio_final}, "
                        f"und_dummy={und_dummy}, gen_dummy={gen_dummy}, audio_dummy={audio_dummy}"
                    )
                else:
                    if "pad" in self.und_token_type:
                        new_item["und_token_indices"] = torch.cat(
                            [new_item["und_token_indices"], padded_indices], dim=1
                        )
                        pad_count["und"] = pad_length
                    elif "pad" in self.gen_token_type:
                        new_item["gen_token_indices"] = torch.cat(
                            [new_item["gen_token_indices"], padded_indices], dim=1
                        )
                        pad_count["gen"] = pad_length
                    elif hasattr(self, "audio_token_type") and "pad" in self.audio_token_type:
                        new_item["audio_token_indices"] = torch.cat(
                            [new_item["audio_token_indices"], padded_indices], dim=1
                        )
                        pad_count["audio"] = pad_length

            # Record token lengths for each branch. Used for modulation vector expansion.
            # If a sample has no tokens for some branch, its token length for that branch is considered as 0,
            # and the 0 length should not be recorded in und/gen/audio_token_lengths, because the corresponding
            # timestep is also non-existent.
            und_token_lengths = []
            gen_token_lengths = []
            audio_token_lengths = [] if "audio_token_indices" not in new_item else []
            for item in items:
                assert item["und_token_indices"].shape[0] > 0, \
                    f"und_token_indices length should be greater than 0, but got {item['und_token_indices'].shape[0]}"
                und_token_lengths.append(item["und_token_indices"].shape[0])
                if item["gen_token_indices"].shape[0] > 0:
                    gen_token_lengths.append(item["gen_token_indices"].shape[0])
                if item["audio_token_indices"].shape[0] > 0:
                    audio_token_lengths.append(item["audio_token_indices"].shape[0])
            # Add pad length to the length of last item
            und_token_lengths[-1] += pad_count["und"]
            new_item["und_token_lengths"] = torch.tensor(und_token_lengths)
            if len(gen_token_lengths) == 0:
                # If no sample has gen tokens, we will add a dummy token as a dummy sample, so we need add a 0 to
                # serve as the placeholder for the dummy sample.
                gen_token_lengths.append(0)
            gen_token_lengths[-1] += pad_count["gen"]
            new_item["gen_token_lengths"] = torch.tensor(gen_token_lengths)
            if len(audio_token_lengths) == 0:
                # The same as gen_token_lengths.
                audio_token_lengths.append(0)
            audio_token_lengths[-1] += pad_count["audio"]
            new_item["audio_token_lengths"] = torch.tensor(audio_token_lengths)

        new_item["pad_count"] = pad_count

        return new_item
