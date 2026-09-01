import os.path as osp
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from index_kits.sampler import DistributedSamplerWithStartIndex
from hymm.models import TokenizerWrapper


class VisuaAudioDataset(Dataset):
    def __init__(
            self, 
            parquet_file, 
            clip_root, 
            clip_pad_emb,
            t5_root, 
            t5_pad_emb, 
            token_root, 
            text_tokenizer, 
            logger=None, 
            max_frames=40, 
            max_text_length=227,
            audio_token_offset=0,
        ):
        if logger is None:
            from loguru import logger
        self.logger = logger
        self.data_list = pd.read_parquet(parquet_file)["file_path"].to_list()
        self.clip_root = clip_root
        self.clip_pad_emb = torch.load(clip_pad_emb, "cpu")[0].unsqueeze(0).repeat(max_frames, 1)
        self.t5_root = t5_root
        self.t5_pad_emb = torch.load(t5_pad_emb, "cpu").unsqueeze(0).repeat(max_text_length, 1)
        self.token_root = token_root
        self.tokenizer = TokenizerWrapper(text_tokenizer, self.logger)
        self.max_frames = max_frames
        self.max_text_length = max_text_length
        self.audio_token_offset = audio_token_offset
    
    def __getitem__(self, idx):
        audio_path = self.data_list[idx]
        base_name = osp.splitext(osp.basename(audio_path))[0]
        clip_path = osp.join(self.clip_root, base_name + ".pt")
        t5_path = osp.join(self.t5_root, base_name + ".pt")
        token_path = osp.join(self.token_root, base_name + ".pt")
        # for t5 and clip, we need to pad to max_length
        clip_embedding = torch.load(clip_path, "cpu")
        if clip_embedding.shape[0] > self.max_frames:
            clip_embedding = clip_embedding[:self.max_frames]
        padeed_clip_embedding = self.clip_pad_emb.clone()
        padeed_clip_embedding[:clip_embedding.shape[0]] = clip_embedding
        t5_embedding = torch.load(t5_path, "cpu")
        if t5_embedding.shape[0] > self.max_text_length:
            t5_embedding = t5_embedding[:self.max_text_length]
        padded_t5_embedding = self.t5_pad_emb.clone()
        padded_t5_embedding[:t5_embedding.shape[0]] = t5_embedding
        # prepcoess tokens into a sequence of 1025 length (minus one for input and target to get 1024)
        audio_tokens = torch.load(token_path, "cpu")["codes"].squeeze() + self.audio_token_offset
        input_tokens = self.tokenizer.encode_visual_t5_audio_sequence(audio_tokens)
        target_token = input_tokens.clone()
        target_token[:-754] = -100      # TODO: make it configurable
        target_token[target_token == self.tokenizer.special_token_map["<pad>"]] = -100
        return {
            "clip_embedding": padeed_clip_embedding, 
            "t5_embedding": padded_t5_embedding, 
            "tokens": input_tokens, 
            "target": target_token,
        }

    def __len__(self):
        return len(self.data_list)


def build_visual_audio_dataloader(cfg):
    dataset = VisuaAudioDataset(
        parquet_file=cfg.parquet_file,
        clip_root=cfg.clip_root,
        clip_pad_emb=cfg.clip_pad_emb,
        t5_root=cfg.t5_root,
        t5_pad_emb=cfg.t5_pad_emb,
        token_root=cfg.token_root,
        text_tokenizer=cfg.text_tokenizer,
        audio_token_offset=cfg.audio_token_offset,
    )
    sampler = DistributedSamplerWithStartIndex(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_params["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )
    return dataloader, sampler
