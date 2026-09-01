import json
import time
import datetime
import math
from typing import Any, Dict
from einops import rearrange
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import os
import torch
import torch.distributed as dist
import torchaudio
import os.path as osp
import subprocess
from transformers import AutoTokenizer, AutoModelForTextEncoding, CLIPProcessor, CLIPModel
from hymm.data_kits.datasets import load_dataset
from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.samplers.logits_processor import get_logits_processors, update_logits_processor_kwargs
from hymm.config import parse_eval_initial_args
from hymm.samplers.base_sampler import BaseSampler
from hymm.models import TokenizerWrapper, AudioTokenizerWrapper
from hymm.utils.file_utils import rank0_logger, safe_file, safe_save_file, save_to_json
from hymm.utils.helpers import default_dtype
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.models.audio_encoders import dac
from hymm.constants import AUDIOSET, VGGSOUND
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class VisualText2AudioARSampler(BaseSampler):
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
        self.audio_token_offset = args.audio_token_offset
        self.cfg_enabled = False
        for logits_processor in args.logits_processors_cfg:
            for name, _ in logits_processor.items():
                if name == "CfgLogitsWarper":
                    self.cfg_enabled = True
    
    @staticmethod
    def build_extra_model(args, model_dict, factor_kwargs, logger=None):
        device = factor_kwargs["device"]
        vae = dac.DAC.load(args.dac_ckpt).to(device)
        model_dict["vae"] = vae

        # =========================== Build text tokenizer ========================
        use_custom_tokenizer = args.get("use_custom_tokenizer", False)
        if use_custom_tokenizer:
            audio_vocab_size = args.dac_voca_size * 2 if args.double_codebook else args.dac_voca_size
            tokenizer = AudioTokenizerWrapper(audio_vocab_size=audio_vocab_size)
        else:
            tokenizer = TokenizerWrapper(args.text_tokenizer, logger)
        model_dict["tokenizer"] = tokenizer

        # =========================== Build logits_processor ======================
        logits_processors = get_logits_processors(args.logits_processors_cfg)
        t5_pad_emb = torch.load(args.t5_pad_emb, "cpu").unsqueeze(0).repeat(args.max_text_length, 1).unsqueeze(0)
        clip_pad_emb = torch.load(args.clip_pad_emb, "cpu")[0].unsqueeze(0).repeat(args.max_frames, 1).unsqueeze(0)

        if args.csv:
            t5_tokenizer = AutoTokenizer.from_pretrained(args.t5_path)
            t5_model = AutoModelForTextEncoding.from_pretrained(args.t5_path).eval().to(device)

            clip_model = CLIPModel.from_pretrained(args.clip_path).eval().to(device)
            clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
        else:
            t5_tokenizer = None
            t5_model = None
            clip_model = None
            clip_processor = None
            
        model_dict["logits_processor"] = logits_processors
        model_dict["t5_pad_emb"] = t5_pad_emb
        model_dict["clip_pad_emb"] = clip_pad_emb
        model_dict["t5_tokenizer"] = t5_tokenizer
        model_dict["t5_model"] = t5_model
        model_dict["clip_model"] = clip_model
        model_dict["clip_processor"] = clip_processor
        return model_dict

    @torch.no_grad()
    def predict_raw(self, video, caption, top_k=None, top_p=None, guidance_scale=None):
        self.model_dict["logits_processor"] = update_logits_processor_kwargs(
            self.model_dict["logits_processor"], 
            top_k=top_k, 
            top_p=top_p, 
            guidance_scale=guidance_scale,
        )
        self.logger.info(self.model_dict["logits_processor"])
        fps = 4
        from decord import VideoReader
        self.logger.info(f"Sample {video}\t{caption}")
        video_reader = VideoReader(video)
        num_frames = len(video_reader)
        step = max(1, num_frames // (fps * (num_frames // video_reader.get_avg_fps())))
        frames =[video_reader[i].asnumpy() for i in range(0, num_frames, int(step))]
        if self.args.repeat_to_full:
            original_length = len(frames)
            while len(frames) < self.args.max_frames:
                frames += frames
            frames = frames[:self.args.max_frames]
        else:
            original_length = 0
        self.logger.info(f"Number of frames: {len(frames)}/{num_frames}/fps{video_reader.get_avg_fps()}")
        clip_inputs = self.model_dict["clip_processor"](images=frames, return_tensors="pt").to(self.device)
        clip_feat = self.model_dict["clip_model"].get_image_features(**clip_inputs).unsqueeze(0)
        t5_inputs = self.model_dict["t5_tokenizer"](caption, return_tensors="pt").to(self.device)
        t5_feat = self.model_dict["t5_model"](**t5_inputs).last_hidden_state
        frames = clip_feat.shape[1]
        audio_token_len = int(self.args.audio_token_max_length/self.args.max_frames*frames)
        original_token_len = int(self.args.audio_token_max_length/self.args.max_frames*original_length)
        if self.args.double_codebook:
            audio_token_len = audio_token_len * 2
        self.logger.info(f"Input for predict: {clip_feat.shape}/{t5_feat.shape}/{audio_token_len}")
        return self.predict("", clip_feat, t5_feat, audio_token_len, original_token_len, show_progress=True)

    @torch.no_grad()
    def predict(self, prompt, clip_feat, t5_feat, audio_token_len, original_token_len, **kwargs):
        show_progress = kwargs.get("show_progress", False)
        tokenizer = self.model_dict["tokenizer"]
        logits_processors = self.model_dict["logits_processor"]
        if self.args.get("no_text", False):
            self.logger.info("Without text")
            prefix_token = tokenizer.encode_visual_sequence_no_text_for_infer(
                max_clip_frame=self.args.max_frames, max_t5_len=self.args.max_text_length
            ).to(self.device).unsqueeze(0)
        else:
            self.logger.info("With text")
            prefix_token = tokenizer.encode_visual_t5_for_audio_infer().to(self.device).unsqueeze(0)
        padded_clip_feat = self.model_dict["clip_pad_emb"].clone().to(self.device)
        if clip_feat.shape[1] > self.args.max_frames:
            clip_feat = clip_feat[:,:self.args.max_frames]
        padded_clip_feat[:, :clip_feat.shape[1]] = clip_feat

        padded_t5_feat = self.model_dict["t5_pad_emb"].clone().to(self.device)
        if t5_feat.shape[1] > self.args.max_text_length:
            t5_feat = t5_feat[:,:self.args.max_text_length]
        padded_t5_feat[:, :t5_feat.shape[1]] = t5_feat
        if self.cfg_enabled:
            prefix_token = torch.cat([prefix_token, prefix_token], dim=0)
            uncond_t5 = self.model_dict["t5_pad_emb"].clone().to(self.device)
            uncond_clip = self.model_dict["clip_pad_emb"].clone().to(self.device)
            t5_feat = torch.cat([padded_t5_feat, uncond_t5], dim=0)
            clip_feat = torch.cat([padded_clip_feat, uncond_clip], dim=0)

        self.model_dict["model"].set_kv_cache(batch_size=prefix_token.shape[0], device=self.device)
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_enabled):
            audio_token = self.model_dict["model"].generate(
                clip_feat, 
                t5_feat,
                prefix_token, 
                audio_token_len, 
                logits_processors, 
                self.device, 
                dac_vocab=self.args.dac_voca_size, 
                audio_token_offset=self.args.audio_token_offset,
                double_vocab=self.args.double_codebook, 
                slice_manualy=self.args.slice_manualy,
                codebook_arrangement=self.args.codebook_arrangement,
                show_progress=show_progress
            )
        self.logger.info("audio_token: {}".format(audio_token.shape))
        # x = dac.DACFile(codes=audio_token.unsqueeze(0), **metainfo)
        if self.args.double_codebook:
            if self.args.codebook_arrangement == "interleave":
                token_book_one = audio_token[0, 0::2]
                token_book_two = audio_token[0, 1::2]
                audio_token = torch.cat([token_book_one.unsqueeze(0), token_book_two.unsqueeze(0)], dim=0)
            else:
                audio_token = audio_token.reshape(2, -1)
            self.logger.info(audio_token.shape)
            self.logger.info(f"{audio_token.min()}, {audio_token.max()}")
            if self.args.repeat_to_full:
                audio_token = audio_token[:, :original_token_len]
        z = self.model_dict["vae"].quantizer.from_codes(audio_token.unsqueeze(0))[0]
        y = self.model_dict["vae"].decode(z)
        y = y.to('cpu')
        return y, self.model_dict["vae"].sample_rate, audio_token
        # if self.args.double_codebook:
        #     z1 = self.model_dict["vae"].quantizer.from_codes(audio_token[0].unsqueeze(0).unsqueeze(0))[0]
        #     y1 = self.model_dict["vae"].decode(z1)
        #     y1 = y1.squeeze(0).to('cpu')
        #     return (y1, y, ), self.model_dict["vae"].sample_rate, audio_token
        # else:
        #     return y, self.model_dict["vae"].sample_rate, audio_token

    @staticmethod
    def make_dirs(dir_root):
        if not osp.exists(dir_root):
            os.makedirs(dir_root, exist_ok=True)

    def save_auido_and_video(self, waveform, sample_rate, filenames, eval_root, output_root):
        audio_root = osp.join(output_root, "audio")
        video_root = osp.join(output_root, "video")
        self.make_dirs(audio_root)
        self.make_dirs(video_root)
        audio_path = []
        for idx, fname in enumerate(filenames):
            audio_file = osp.join(audio_root, f"{fname}.wav")
            video_file = osp.join(video_root, f"{fname}.mp4")
            torchaudio.save(audio_file, waveform[idx], sample_rate)
            audio_path.append(audio_file)
            original_video = osp.join(eval_root, "videos", fname + ".mp4")
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                original_video,
                "-i",
                audio_file,
                "-c:v",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                video_file
            ]
            process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # _, stderr = process.communicate()
        return audio_path


    @torch.no_grad()
    def eval(self, batch_size, save_path, save_base=None, extra_save_info=None, **kwargs):
        if not self.score_models_loaded:
            for metric_name, metric in self.metric_dict.items():
                metric.load_model(self.logger)
            self.score_models_loaded = True

            # Check batch_size, and convert it to a single integer
        if isinstance(batch_size, (list, tuple)):
            assert len(batch_size) == 1, f"Expected a single batch size, got {batch_size}"
            batch_size = batch_size[0]

        # Check extra_save_info to avoid saving error and convert it to a dictionary
        extra_save_info = default_dtype(extra_save_info, {})

        # Load datasets. A dataset is valid if its max_size is greater than or equal to the current target size.
        # Remove duplicates for saving computation.
        valid_dataset_names = sorted(list(set([ metric.dataset_name for metric_name, metric in self.metric_dict.items()])))
        dataloaders = []
        for dataset_name in valid_dataset_names:
            dataset = load_dataset(dataset_name)
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"{dataset_name} dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, dataset_name))

        # Do prediction
        for dataloader, dataset, dataset_name in dataloaders:
            if dataset_name == "audioset":
                eval_root = AUDIOSET
            elif dataset_name == "vggsound":
                eval_root = VGGSOUND
            total_batches = len(dataloader)
            self.logger.info(
                f"********************************** "
                f"Evaluation on {dataset_name}, in {eval_root}, Total batches: {total_batches} "
                f"**********************************"
            )
            if save_base is not None:
                if "{}" in str(save_base):
                    save_dir = Path(str(save_base).format(dataset_name))
                else:
                    save_dir = Path(str(save_base) + f"{dataset_name}")
                self.logger.info(f"{dataset_name} audios&videos will be saved to {save_dir}")
            else:
                save_dir = self.args.sample_save_path

            for batch_index, batch in enumerate(dataloader, start=1):
                batch: Dict[str, Any]
                self.logger.info(f"Batch {batch_index}/{total_batches}")
                t5_feat = batch["t5_feat"].to(self.device)
                clip_feat = batch["clip_feat"].to(self.device)
                audio_token_len = 750 * 2       # TODO: support configurable
                wave_form, sample_rate, token = self.predict(
                    prompt="",
                    t5_feat=t5_feat,
                    clip_feat=clip_feat,
                    audio_token_len=audio_token_len,
                    original_token_len=0,
                )
                audio_path = self.save_auido_and_video(
                    wave_form, 
                    sample_rate, 
                    batch["path"], 
                    eval_root, 
                    save_dir
                )
                
                proc_inputs = dict()
                proc_inputs["prompts"] = batch['prompt']
                proc_inputs["audio_paths"] = audio_path
                proc_inputs["video_paths"] = [osp.join(eval_root, "videos", f"{path}.mp4") for path in batch["path"]]

                start_time = time.time()
                for metric_name, metric in self.metric_dict.items():
                    if metric.dataset_name == dataset_name:
                        results = metric.process(**proc_inputs)
                        self.logger.info(f"Metric {metric_name} results: {results}")
                        self.valid_metrics.add(metric_name)
                gen_time = time.time() - start_time
                self.logger.info(f"Process time: {gen_time}")

        self.logger.info("All dataloaders are finished, begin to gather results")
        dist.barrier()
        metric_gather_results = {}
        for metric_name, metric in self.metric_dict.items():
            if metric_name in self.valid_metrics:
                metric_gather_results[metric_name] = metric.all_gather_results()

        results = []
        if self.rank == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            for metric_name, metric in self.metric_dict.items():
                if metric_name in self.valid_metrics:
                    output = metric.compute_metrics(metric_gather_results[metric_name])
                    if isinstance(output, dict):
                        for key, (value, count) in output.items():
                            results.append({
                                "metric": f"{metric_name}_{key}",
                                "value": value,
                                "count": count,
                                "timestamp": timestamp,
                                "extra": extra_save_info,
                            })
                    elif isinstance(output, tuple):
                        value, count = output
                        results.append({
                            "metric": metric_name,
                            "value": value,
                            "count": count,
                            "timestamp": timestamp,
                            "extra": extra_save_info,
                        })
                    else:
                        raise ValueError(
                            f"Output of metric {metric_name} should be a dict or a two-value tuple, but got {output}"
                        )
            self.logger.info(results)
            accumulated_results = results[:]
            save_path = safe_file(save_path)
            if save_path.exists():
                with open(save_path, "r") as f:
                    ori_results = json.load(f)
                accumulated_results = ori_results + results
            save_to = safe_save_file(save_path, accumulated_results, save_fn=save_to_json)
            self.logger.info(f"Evaluation results saved to {save_to}")

        # gather finish, reset
        for metric_name, metric in self.metric_dict.items():
            metric.reset()
        self.valid_metrics.clear()

        # Only rank-0 returns the valid results
        return results

def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)
    logger.info(f"World size: {world_size}, rank: {rank}, device: {device}")
    sampler = VisualText2AudioARSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args
    logger.info(args.sample_save_path)
    if not os.path.exists(args.sample_save_path):
        os.makedirs(args.sample_save_path, exist_ok=True)
    video_out_dir = os.path.join(args.sample_save_path, "videos")
    if not os.path.exists(video_out_dir):
        os.makedirs(video_out_dir, exist_ok=True)
    audio_out_dir = os.path.join(args.sample_save_path, "audios")
    if not os.path.exists(audio_out_dir):
        os.makedirs(audio_out_dir, exist_ok=True)

    # Start evaluation
    if args.interactive:
        while True:
            if args.prompt is None:
                # Ask for the next prompt
                inputs = input("Input prompt (`q` to quit): ")
                if inputs == "q":
                    break
                prompt = inputs
            else:
                prompt = args.prompt
                args.prompt = None

            # Determine the seed
            if args.seed_type in ["auto", "fixed"]:
                seed = args.seed
            elif args.seed_type == "random":
                seed = None
            else:
                raise ValueError(
                    f"When evaluating `prompt`, `seed_type` must be one of ['auto', 'fixed', 'random'], "
                    f"got {args.seed_type}."
                )
            # Start sampling
            outputs = sampler.predict(
                prompt=prompt,
                size=args.image_size,
                seed=seed,
                verbose=1,
            )
            samples = outputs["samples"]
            # Save the generated images
            save_paths = sampler.get_default_sample_save_paths(len(samples), prompt, save_dir=args.sample_save_path)
            sampler.save_batch_image(samples, save_paths)
            logger.info(f"Save the generated image to: {save_paths}")

    elif args.csv:
        val_data = pd.read_csv(args.csv)
        logger.info(f"Total number of eval samples: {len(val_data)}")
        current_rank_data = np.array_split(val_data, world_size)[rank]
        logger.info(f"Total number of eval samples of rank {rank}: {len(current_rank_data)}")
        for idx, data in current_rank_data.iterrows():
            save_base_name = osp.splitext(data["video"])[0].split("/")[0]
            save_base_name = f"{idx:04d}"
            original_video = osp.join(args.eval_data_root, data["video"])
            caption = data["structure_caption"]
            wave_form, sample_rate, token = sampler.predict_raw(original_video, caption)
            
            outputs_video = osp.join(video_out_dir, f"{save_base_name}.mp4")
            if isinstance(wave_form, tuple):
                outputs_file = osp.join(audio_out_dir, f"{save_base_name}_0.wav")
                torchaudio.save(outputs_file, wave_form[0], sample_rate)
                outputs_file = osp.join(audio_out_dir, f"{save_base_name}_all.wav")
                torchaudio.save(outputs_file, wave_form[1], sample_rate)
            else:
                outputs_file = osp.join(audio_out_dir, f"{save_base_name}.wav")
                torchaudio.save(outputs_file, wave_form.squeeze(0), sample_rate)
            logger.info(f"Waveform save to:\t{outputs_file} with sample rate {sample_rate}")
            torch.save(token.cpu(), osp.join(audio_out_dir, f"{save_base_name}.pt"))
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
                outputs_video
            ]
            process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Merge audio error {stderr}")
            else:
                logger.info(f"Video save to:\t{outputs_video}")
    elif args.parquet:
        val_data = pd.read_parquet(args.parquet)
        logger.info(f"Total number of eval samples: {len(val_data)}")
        current_rank_data = np.array_split(val_data, world_size)[rank]
        logger.info(f"Total number of eval samples of rank {rank}: {len(current_rank_data)}")
        for idx, data in current_rank_data.iterrows():
            path = data["path"]
            caption = data["structure_caption"]
            logger.info(f"Sample {idx}/{len(current_rank_data)}\t{path}\t{caption}")
            t5_feat = osp.join(args.eval_data_root, "t5_feat", path + ".pt")
            clip_feat = osp.join(args.eval_data_root, "clip", path + ".pt")
            t5_feat = torch.load(t5_feat, "cpu").to(device).unsqueeze(0)
            clip_feat = torch.load(clip_feat, "cpu").to(device).unsqueeze(0)
            frames = clip_feat.shape[1]
            audio_token_len = int(args.audio_token_max_length/args.max_frames*frames)
            if args.double_codebook:
                audio_token_len = audio_token_len * 2
            wave_form, sample_rate, token = sampler.predict(
                "",
                clip_feat=clip_feat,
                t5_feat=t5_feat,
                audio_token_len=audio_token_len,
                original_token_len=0
            )
            outputs_file = osp.join(audio_out_dir, f"{idx:04d}.wav")
            outputs_video = osp.join(video_out_dir, f"{idx:04d}.mp4")
            logger.info(f"Waveform save to:\t{outputs_file} with sample rate {sample_rate}")
            if isinstance(wave_form, tuple):
                torchaudio.save(outputs_file, wave_form[1], sample_rate)
                output_file_0 = osp.join(audio_out_dir, f"{idx:04d}_0.wav")
                torchaudio.save(output_file_0, wave_form[0], sample_rate)
            else:
                torchaudio.save(outputs_file, wave_form.squeeze(0), sample_rate)
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
                outputs_video
            ]
            process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Merge audio error {stderr}")
            else:
                logger.info(f"Video save to:\t{outputs_video}")
                        

if __name__ == "__main__":
    main()
