import os
import os.path as osp
import subprocess
from fractions import Fraction
from io import BytesIO
from typing import List, Tuple

import av
import nltk
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from decord import VideoReader
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import v2
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import Normalize

from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.config import parse_eval_initial_args
from hymm.samplers.base_sampler import BaseSampler
from hymm.utils.file_utils import rank0_logger
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.diffusion import load_diffusion_pipeline
from hymm.trainers.textvisual2audio_diffusion_trainer import patch_clip
from hymm.models.visual_encoders.imagebind import data
from hymm.models.visual_encoders.imagebind import imagebind_model
from hymm.models.visual_encoders.synchformer import Synchformer


FPS_VISUAL = {"clip": 8, "cavp": 8, "synchformer": 25}
SYNCHFORMER_CKPT_PATH = "/apdcephfs_gy2/share_302507476/1_public_models/MMAudio/ext_weights/synchformer_state_dict.pth"
IMAGEBIND_CKPT_PATH = "/apdcephfs_gy2/share_302507476/yutaocui/model_zoo/imagebind_huge.pth"


class VisualText2AudioDiffSampler(BaseSampler):
    def __init__(self, args, model_dict, ckpt_path=None, rank=0, world_size=1, device=0, logger=None):
        super().__init__(
            args=args,
            model_dict=model_dict,
            ckpt_path=ckpt_path,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )
        self.model_dtype = PRECISION_TO_TYPE[args.precision]
        self.autocast_enabled = self.model_dtype in [torch.half, torch.bfloat16]

        # Build visual condition models
        self.visual_feat_type = getattr(self.args, "visual_feat_type", "clip")
        # sync modulation type: 'none' or 'synchformer'
        self.sync_modulation_type = getattr(self.args, "sync_modulation_type", "none")
        if "synchformer" in [self.visual_feat_type, self.sync_modulation_type]:
            self.synchformer = Synchformer().cuda().eval()
            self.synchformer.load_state_dict(torch.load(SYNCHFORMER_CKPT_PATH, weights_only=True, map_location="cpu"))
            self.sync_transforms = v2.Compose(
                [
                    v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
                    v2.CenterCrop(224),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        self.logger.info(
            f"visual feat type: {self.visual_feat_type}, sync modulation type: {self.sync_modulation_type}"
        )

        # Build imagebind, which is used for re-rank
        self.extra_imagebind_cond = True
        if self.extra_imagebind_cond:
            self.ib_model = imagebind_model.imagebind_huge(pretrained=True, ckpt_path=IMAGEBIND_CKPT_PATH).to("cuda")
            self.ib_model.eval()

        # Load diffusion pipeline
        self.pipeline = load_diffusion_pipeline(
            args=self.args,
            rank=self.rank,
            pipeline_name="textvisual2audiodacvae",
            diffusion_model=self.model_dict["model"],
            device=self.device,
        )

        # 使用clip提取text feature
        if self.args.use_clip_text_feat:
            self.tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
            self.clip_model = create_model_from_pretrained(
                "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", return_transform=False
            ).eval()
            self.clip_model = patch_clip(self.clip_model).to(self.device)

        # Prepare things for clip-huge
        self.clip_feat_type = getattr(self.args, "clip_feat_type", "large")
        if self.clip_feat_type == "huge":
            self.clip_preprocess = Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
            self.clip_transform = v2.Compose(
                [
                    v2.Resize((384, 384), interpolation=v2.InterpolationMode.BICUBIC),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )

    @torch.inference_mode()
    def encode_text_clip(self, text: List[str], device) -> torch.Tensor:
        assert self.clip_model is not None, "CLIP is not loaded"
        assert self.tokenizer is not None, "Tokenizer is not loaded"
        # x: (B, L)
        tokens = self.tokenizer(text).to(device)
        return self.clip_model.encode_text(tokens, normalize=True)

    @staticmethod
    def build_extra_model(args, model_dict, factor_kwargs, logger=None):
        device = factor_kwargs["device"]

        clip_feat_type = getattr(args, "clip_feat_type", "large")
        if clip_feat_type == "large":
            clip_model = CLIPModel.from_pretrained(args.clip_path).eval().to(device)
            clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
            model_dict["clip_model"] = clip_model
            model_dict["clip_processor"] = clip_processor

        return model_dict

    @torch.inference_mode()
    def encode_video_with_clip_mmaudio(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.clip_model is not None, "CLIP is not loaded"
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        assert c == 3 and h == 384 and w == 384
        x = self.clip_preprocess(x)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            outputs.append(self.clip_model.encode_image(x[i : i + batch_size], normalize=True))
        x = torch.cat(outputs, dim=0)
        # x = self.clip_model.encode_image(x, normalize=True)
        x = rearrange(x, "(b t) d -> b t d", b=b)
        return x

    @torch.inference_mode()
    def encode_video_with_sync(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        """
        The input video of x is best to be in fps of 24 of greater than 24.
        Input:
            x: tensor in shape of [B, T, C, H, W]
            batch_size: the batch_size for synchformer inference
        """
        assert self.synchformer is not None, "Synchformer is not loaded"

        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        # partition the video, 相当于 kernel_size=16, stride=8
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size : i * step_size + segment_size])
        x = torch.stack(segments, dim=1).cuda()  # (B, num_segments, segment_size, 3, 224, 224)

        # import ipdb; ipdb.set_trace()
        outputs = []
        if batch_size < 0:
            batch_size = b * num_segments
        x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
        for i in range(0, b * num_segments, batch_size):
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.half):
                outputs.append(self.synchformer(x[i : i + batch_size]))
        x = torch.cat(outputs, dim=0)  # [b * num_segments, 1, 8, 768]
        x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b)
        return x

    def get_frames_decord(
        self,
        video,
        fps,
        max_length: int = None,
    ):
        video_reader = VideoReader(video)

        num_frames = len(video_reader)
        source_fps = video_reader.get_avg_fps()
        step = source_fps / fps
        vid_len_in_s = num_frames / source_fps
        indices = np.arange(0, num_frames, step)
        indices = np.floor(indices).astype(int)
        indices = np.clip(indices, 0, num_frames - 1)
        # print(num_frames, indices)
        frames = video_reader.get_batch(indices).asnumpy()

        # 按最大长度截断
        if max_length is not None and len(frames) > int(max_length * fps):
            frames = frames[: int(max_length * fps)]
            vid_len_in_s = max_length

        self.logger.info(f"Number of frames: {len(frames)}/{num_frames}/fps{video_reader.get_avg_fps()}")
        return frames, vid_len_in_s

    def get_frames_av(
        self,
        video_path,
        fps,
        max_length: float = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Fraction]:
        end_sec = max_length if max_length is not None else 15
        next_frame_time_for_each_fps = 0.0
        time_delta_for_each_fps = 1 / fps

        all_frames = []
        output_frames = []

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            ori_fps = stream.guessed_rate
            stream.thread_type = "AUTO"
            for packet in container.demux(stream):
                for frame in packet.decode():
                    frame_time = frame.time
                    if frame_time < 0:
                        continue
                    if frame_time > end_sec:
                        break

                    frame_np = None

                    this_time = frame_time
                    while this_time >= next_frame_time_for_each_fps:
                        if frame_np is None:
                            frame_np = frame.to_ndarray(format="rgb24")

                        output_frames.append(frame_np)
                        next_frame_time_for_each_fps += time_delta_for_each_fps

        output_frames = np.stack(output_frames)

        vid_len_in_s = len(output_frames) / fps
        # 按最大长度截断
        if max_length is not None and len(output_frames) > int(max_length * fps):
            output_frames = output_frames[: int(max_length * fps)]
            vid_len_in_s = max_length

        return output_frames, vid_len_in_s

    def get_visual_features(self, visual_feat_types, video, duration_sec=10):
        visual_features = {}

        for visual_feat_type in visual_feat_types:
            if visual_feat_type == "clip":
                # get clip features
                # frames, ori_vid_len_in_s = self.get_frames_decord(video, FPS_VISUAL['clip'])
                if self.clip_feat_type == "large":
                    frames, ori_vid_len_in_s = self.get_frames_av(video, FPS_VISUAL["clip"])
                    clip_inputs = self.model_dict["clip_processor"](images=frames, return_tensors="pt").to(self.device)
                    visual_features[visual_feat_type] = (
                        self.model_dict["clip_model"].get_image_features(**clip_inputs).unsqueeze(0)
                    )  # [1, length * FPS, 768]
                elif self.clip_feat_type == "huge":
                    frames, ori_vid_len_in_s = self.get_frames_av(video, FPS_VISUAL["clip"])
                    images = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
                    clip_frames = self.clip_transform(images).to(self.device).unsqueeze(0)
                    visual_features[visual_feat_type] = self.encode_video_with_clip_mmaudio(
                        clip_frames
                    )  # [1, length * FPS, 768]
            elif visual_feat_type == "synchformer":
                # frames, ori_vid_len_in_s = self.get_frames_decord(video, FPS_VISUAL["synchformer"])
                frames, ori_vid_len_in_s = self.get_frames_av(video, FPS_VISUAL["synchformer"])
                images = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
                sync_frames = self.sync_transforms(images).unsqueeze(0)  # [1, T, 3, 224, 224]
                # [1, num_segments * 8, channel_dim], e.g. [1, 240, 768] for 10s video
                visual_features[visual_feat_type] = self.encode_video_with_sync(sync_frames)
            elif visual_feat_type == "imagebind":
                ib_fps = 25
                frames, _ = self.get_frames_decord(video, fps=ib_fps)
                video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [T, C, H, W]
                duration = video_tensor.shape[0] / ib_fps
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.half):
                    inputs = {
                        imagebind_model.ModalityType.VISION: data.load_and_transform_video_tensor_data(
                            [video_tensor], [duration], "cuda", fps=ib_fps
                        ),
                    }
                    embeddings = self.ib_model(inputs)
                vision_emb = embeddings[imagebind_model.ModalityType.VISION].unsqueeze(1)  # [N, 1, C]
                visual_features[visual_feat_type] = vision_emb
            else:
                raise KeyError(f"Unsupported visual feature type: {visual_feat_type}")

        vid_len_in_s = sync_frames.shape[1] / FPS_VISUAL["synchformer"]
        self.logger.info(f"video len: {ori_vid_len_in_s}, sync_len_in_s: {vid_len_in_s}")

        return visual_features, vid_len_in_s

    @torch.no_grad()
    def predict_raw(self, idx, video, caption, n_samples):
        # fps = 12
        self.logger.info(f"Sample {video}\t{caption}")

        visual_feat_types = [self.visual_feat_type, self.sync_modulation_type]
        if self.extra_imagebind_cond:
            visual_feat_types.append("imagebind")
        visual_feats, audio_len_in_s = self.get_visual_features(visual_feat_types, video, duration_sec=10)

        visual_feat = visual_feats[self.visual_feat_type]
        sync_feat = visual_feats[self.sync_modulation_type]
        imagebind_feat = None
        if self.extra_imagebind_cond:
            imagebind_feat = visual_feats["imagebind"]

        cond_mask = None
        # get text features
        if self.args.use_clip_text_feat:
            caption = ["", caption]
            text_clip_feat, clip_text_mask = self.encode_text_clip(caption, self.device)
            text_feat = text_clip_feat[1:]
            uncond_text_feat = text_clip_feat[:1]
            if self.model_dict["model_settings"].t5_length < text_feat.shape[1]:
                text_seq_length = self.model_dict["model_settings"].t5_length
                text_feat = text_feat[:, :text_seq_length]
                uncond_text_feat = uncond_text_feat[:, :text_seq_length]

        self.logger.info(
            f"Input for predict: text_feat: ({text_feat.shape})/ visual_feat ({self.visual_feat_type}) {visual_feat.shape}/ audio_len_in_s {audio_len_in_s}"
        )
        if sync_feat is not None:
            self.logger.info(f"sync_feat ({self.sync_modulation_type}) {sync_feat.shape}/")
        return self.predict(
            idx,
            visual_feat,
            text_feat,
            audio_len_in_s,
            uncond_text_feat,
            sync_feat,
            imagebind_feat,
            cond_mask,
            n_samples,
        )

    @torch.no_grad()
    def predict(
        self,
        idx,
        clip_feat,
        text_feat,
        audio_len_in_s,
        uncond_text_feat,
        sync_feat,
        imagebind_feat=None,
        cond_mask=None,
        n_samples=1,
        **kwargs,
    ):
        padded_clip_feat = clip_feat
        padded_sync_feat = sync_feat  # [:, :self.max_sync_length]

        guidance_scale = self.args.guidance_scale

        # prepare unconditional clip feature for cfg in Aries
        if guidance_scale > 1 and "TV2A-Aries" in self.args.model_name:
            enable_learnable_empty_visual_feat = getattr(self.args, "enable_learnable_empty_visual_feat", False)
            if enable_learnable_empty_visual_feat:
                uncond_clip_feat = self.pipeline.diffusion_model.get_empty_clip_sequence(
                    bs=1, len=padded_clip_feat.shape[1]
                )
            else:
                uncond_clip_feat = self.model_dict["clip_pad_emb"].clone().to(self.device)
            padded_clip_feat = torch.cat((uncond_clip_feat, padded_clip_feat), dim=0)

            assert enable_learnable_empty_visual_feat
            uncond_sync_feat = self.pipeline.diffusion_model.get_empty_sync_sequence(
                bs=1, len=padded_sync_feat.shape[1]
            )
            padded_sync_feat = torch.cat((uncond_sync_feat, padded_sync_feat), dim=0)

        # prepare n_samples' condition
        padded_clip_feat = padded_clip_feat.repeat(1, n_samples, 1)
        padded_clip_feat = padded_clip_feat.view(padded_clip_feat.shape[0] * n_samples, -1, padded_clip_feat.shape[2])
        padded_sync_feat = padded_sync_feat.repeat(1, n_samples, 1)
        padded_sync_feat = padded_sync_feat.view(padded_sync_feat.shape[0] * n_samples, -1, padded_sync_feat.shape[2])

        # FIXME (yutaocui): config
        vae_sampling_rate = 50
        sampling_rate = 48000

        if cond_mask is None:
            cond_mask = torch.ones((1, text_feat.shape[1]), dtype=torch.bool, device=self.device)

        # ======================================== Generate audio ======================================
        with torch.autocast(device_type="cuda", dtype=self.model_dtype, enabled=self.autocast_enabled):
            model_input_extra_kwargs = dict(
                clip_feat=padded_clip_feat,  # [b, xxx, 768]
                sync_feat=padded_sync_feat,
            )
            wav = self.pipeline(
                audio_length_in_s=audio_len_in_s,
                guidance_scale=guidance_scale,
                prompt_embeds=text_feat,
                prompt_attention_mask=cond_mask,
                negative_prompt_embeds=uncond_text_feat,
                negative_prompt_attention_mask=cond_mask,
                num_inference_steps=50,  # TODO
                max_sequence_length=self.args.max_text_length,
                output_type="tensor",
                model_input_extra_kwargs=model_input_extra_kwargs,
                vae_sampling_rate=vae_sampling_rate,
                num_samples_per_condition=n_samples,
                sampling_rate=sampling_rate,
                return_dict=True,
            ).audios

        # 根据imagebind score进行rerank筛选
        audios_byte = []
        for idx in range(n_samples):
            aud_byte = BytesIO()
            torchaudio.save(aud_byte, wav[idx], sampling_rate, format="wav")
            audios_byte.append(aud_byte.getvalue())
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.half):
            inputs = {
                imagebind_model.ModalityType.AUDIO: data.load_and_transform_audio_data(audios_byte, self.device),
            }
            embeddings = self.ib_model(inputs)
            vision_embs = imagebind_feat.squeeze(1)  # N x C
            audio_embs = embeddings[imagebind_model.ModalityType.AUDIO]  # N x C
        cos_sims = torch.cosine_similarity(vision_embs, audio_embs, dim=1)
        self.logger.info(f"cos_sims: {cos_sims}")
        # 找到cos_sims中的最大值的索引
        max_index = torch.argmax(cos_sims)
        wav = wav[max_index]

        eff_wav = wav
        # eff_wav = wav[..., :int(audio_len_in_s * sampling_rate)]
        self.logger.info("effective audio waveform shape: {}".format(eff_wav.shape))
        return eff_wav, sampling_rate


NLTK_PRE_DATA_PATH = "/apdcephfs_gy2/share_302507476/yutaocui/model_zoo/nltk_data"
MEANINGLESS_WORDS = ["is", "are", "being", "am", "be", "was", "were", "with", "object", "objects", "something"]
nltk.data.path.append(NLTK_PRE_DATA_PATH)


def extract_nouns(caption, meaningless_words=MEANINGLESS_WORDS):
    words = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(words)
    nouns = [word for word, pos in pos_tags if (pos.startswith("NN") and word not in meaningless_words)]
    return nouns


def extract_verbs(caption, meaningless_words=MEANINGLESS_WORDS):
    words = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(words)
    verbs = [word for word, pos in pos_tags if pos.startswith("VB") and word not in meaningless_words]
    return verbs


def main():
    initial_args = parse_eval_initial_args()[0]
    print(initial_args)
    if initial_args.ddp:
        mode = "ddp"
    elif initial_args.deepspeed:
        mode = "deepspeed"
    else:
        mode = "none"
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)
    logger.info(f"World size: {world_size}, rank: {rank}, device: {device}")

    sampler = VisualText2AudioDiffSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args

    if not os.path.exists(args.sample_save_path):
        os.makedirs(args.sample_save_path, exist_ok=True)
    video_out_dir = os.path.join(args.sample_save_path, "videos")
    if not os.path.exists(video_out_dir):
        os.makedirs(video_out_dir, exist_ok=True)
    audio_out_dir = os.path.join(args.sample_save_path, "audios")
    if not os.path.exists(audio_out_dir):
        os.makedirs(audio_out_dir, exist_ok=True)

    if args.csv:
        val_data = pd.read_csv(args.csv)
        current_rank_data = np.array_split(val_data, world_size)[rank]
        logger.info(f"Total number of eval samples of rank {rank}: {len(current_rank_data)}")
        for idx, data in current_rank_data.iterrows():
            original_video = osp.join(args.eval_data_root, data["video"])
            caption = data["SoundCaption"]
            # if args.use_keywords_infer_v2a:
            #     nouns = extract_nouns(caption)
            #     verbs = extract_verbs(caption)
            #     unique_words = set(nouns) | set(verbs)
            #     caption = ", ".join(unique_words)
            # if args.enable_high_quality_tag_infer_v2a:
            #     # caption = caption + ", high-quality"
            #     caption = "high-quality, " + caption
            logger.info(f"Caption: {caption}")
            outputs_file = osp.join(audio_out_dir, f"{idx:04d}.wav")
            outputs_video = osp.join(video_out_dir, f"{idx:04d}.mp4")
            if not os.path.exists(outputs_file) or not os.path.exists(outputs_video):
                wave_form, sample_rate = sampler.predict_raw(idx, original_video, caption, n_samples=5)
                logger.info(f"Waveform save to:\t{outputs_file} with sample rate {sample_rate}")

            if not os.path.exists(outputs_file) or not os.path.exists(outputs_video):
                torchaudio.save(outputs_file, wave_form, sample_rate)
            if not os.path.exists(outputs_video) or not os.path.exists(outputs_video):
                ffmpeg_command = [
                    "ffmpeg",
                    "-i",
                    original_video,
                    "-i",
                    outputs_file,
                    "-c:v",
                    "copy",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    outputs_video,
                ]
                process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                _, stderr = process.communicate()
                if process.returncode != 0:
                    logger.error(f"Merge audio error {stderr}")
                else:
                    logger.info(f"Video save to:\t{outputs_video}")
    elif args.parquet:
        # vggsound parquet
        val_data = pd.read_parquet(
            "/apdcephfs_gy2/share_302507476/1_public_models/hymm_ar_assets/evaluation/dataset/vggsound/val/val.parquet"
        )
        audio_root = (
            "/apdcephfs_gy2/share_302507476/1_public_models/hymm_ar_assets/evaluation/dataset/vggsound/val/videos"
        )

        logger.info(f"Total number of eval samples: {len(val_data)}")
        current_rank_data = np.array_split(val_data, world_size)[rank]
        logger.info(f"Total number of eval samples of rank {rank}: {len(current_rank_data)}")
        for idx, data in current_rank_data.iterrows():
            path = f"{data['path']}"
            video_path = osp.join(audio_root, f"{path}.mp4")
            caption = data["caption_genau"]

            wave_form, sample_rate = sampler.predict_raw(idx, video_path, caption)
            outputs_file = osp.join(audio_out_dir, f"{path}.wav")
            outputs_video = osp.join(video_out_dir, f"{path}.mp4")
            logger.info(f"Waveform save to:\t{outputs_file} with sample rate {sample_rate}")
            torchaudio.save(outputs_file, wave_form, sample_rate)
            original_video = osp.join(args.eval_data_root, "videos", path + ".mp4")
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                original_video,
                "-i",
                outputs_file,
                "-c:v",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                outputs_video,
            ]
            process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Merge audio error {stderr}")
            else:
                logger.info(f"Video save to:\t{outputs_video}")


if __name__ == "__main__":
    main()
