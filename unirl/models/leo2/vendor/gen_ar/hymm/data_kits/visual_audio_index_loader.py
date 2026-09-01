import numpy as np
import random
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from index_kits import ArrowIndexV2, MultiIndexV2, arrow_mapper
from index_kits.sampler import DistributedSamplerWithStartIndex
from hymm.models import TokenizerWrapper, AudioTokenizerWrapper
from hymm.utils.torch_utils import set_worker_seed_builder
from hymm.data_kits.index_dataset import IndexDataset


class VisualAudioIndexDataset(IndexDataset):
    def __init__(
        self,
        index_file,
        text_tokenizer,
        clip_pad_emb,
        t5_pad_emb,
        audio_token_offset=0,
        index_kwargs=None,
        logger=None,
        use_t5=True,
        T5_dim=4096,
        clip_dim=768,
        max_frames=40, 
        max_text_length=227,
        audio_token_max_length=750,
        max_sequence_len=1025,
        debug=False,
        drop_clip=0.0,
        drop_text=0.0,
        feat_in_shadow=False,
        shadow_file_cfg={},
        double_codebook=False,
        dac_voca_size=4096,
        codebook_arrangement="interleave",
        use_first_code_book=False,
        no_text=False,
        use_custom_tokenizer=False,
    ):
        if logger is None:
            from loguru import logger
            self.logger = logger
        # Prepare index manager
        self.index_file = index_file
        index_load_kwargs = dict(
            batch_size=index_kwargs.get("batch_size", 1),
            world_size=index_kwargs.get("world_size", 1),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
        )
        self.feat_in_shadow = feat_in_shadow
        shadow_file_fn = {}
        if feat_in_shadow:
            self.logger.info(shadow_file_cfg)
            shadow_file_fn.update({"clip": partial(arrow_mapper, suffix=shadow_file_cfg["clip_suffix"])})
            shadow_file_fn.update({"t5": partial(arrow_mapper, suffix=shadow_file_cfg["t5_suffix"])})
            shadow_file_fn.update({"dac": partial(arrow_mapper, suffix=shadow_file_cfg["dac_suffix"])})
            self.clip_col = shadow_file_cfg["clip_col"]
            self.t5_col = shadow_file_cfg["t5_col"]
            self.dac_col = shadow_file_cfg["dac_col"]
        index_load_kwargs["shadow_file_fn"] = shadow_file_fn
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"Using {self.index_manager}")
        self.audio_token_offset = audio_token_offset
        self.double_codebook = double_codebook
        self.use_first_code_book = use_first_code_book
        self.codebook_arrangement = codebook_arrangement
        self.dac_voca_size = dac_voca_size
        self.drop_clip = drop_clip
        self.drop_text = drop_text
        self.debug = debug
        self.t5_dim = T5_dim
        self.clip_dim = clip_dim
        self.max_frames = max_frames
        self.audio_token_max_length = audio_token_max_length
        self.max_sequence_len = max_sequence_len
        self.max_text_length = max_text_length
        self.clip_pad_emb = torch.load(clip_pad_emb, "cpu")[0].unsqueeze(0).repeat(max_frames, 1)
        self.t5_pad_emb = torch.load(t5_pad_emb, "cpu").unsqueeze(0).repeat(max_text_length, 1)
        if use_custom_tokenizer:
            audio_vocab_size = self.dac_voca_size * 2 if double_codebook else self.dac_voca_size
            self.tokenizer = AudioTokenizerWrapper(audio_vocab_size)
        else:
            self.tokenizer = TokenizerWrapper(text_tokenizer, self.logger)
        self.use_t5 = use_t5
        self.no_text = no_text
        self.logger.info(
            f"Auido dataset info:\n"
            f"max_frames: {self.max_frames},\n"
            f"max_text_length: {self.max_text_length},\n"
            f"audio_token_max_length: {self.audio_token_max_length},\n"
            f"max_sequence_len: {self.max_sequence_len},\n"
            f"double_codebook: {self.double_codebook},\n"
            f"dac_voca_size: {self.dac_voca_size},\n"
            f"codebook_arrangement: {self.codebook_arrangement},\n"
            f"drop_clip: {self.drop_clip}, \n"
            f"drop_text: {self.drop_text}, \n"
            f"use_t5: {self.use_t5}, \n"
            f"no_text: {self.no_text}"
        )

    def load_index(
        self,
        batch_size,
        world_size,
        shadow_file_fn={},
        sample_strategy="uniform",
        probability=None,
    ):
        if isinstance(self.index_file, str):
            self.index_file = [self.index_file]
        assert isinstance(self.index_file, (list, tuple)), (
            f"`index_file` should be a str or a list of str, got {type(self.index_file)}"
        )
        self.logger.info(f"Loading dataset index: {self.index_file}")
        self.logger.info(f"shadow_file_fn: {shadow_file_fn}")
        index_file = self.index_file
        if len(index_file) == 1:
            index_manager = ArrowIndexV2(
                index_file[0],
                shadow_file_fn=shadow_file_fn,
            )
        else:
            index_manager = MultiIndexV2(
                index_file,
                shadow_file_fn=shadow_file_fn,
                sample_strategy=sample_strategy,
                probability=probability,
            )
        return index_manager

    def get_data_from_shadow(self, index):
        clip_feat = self.index_manager.get_attribute(index, column="clip_feat", shadow="clip")
        t5_feat = self.index_manager.get_attribute(index, column="t5_feat", shadow="t5")
        dac_token = self.index_manager.get_attribute(index, column="dac_token", shadow="dac")
        return clip_feat, t5_feat, dac_token
    
    def __getitem__(self, index):
        if self.feat_in_shadow:
            clip_feat, t5_feat, dac_token = self.get_data_from_shadow(index)
        else:
            clip_feat = self.index_manager.get_attribute(index, column="clip_feat")
            t5_feat = self.index_manager.get_attribute(index, column="t5_feat")
            dac_token = self.index_manager.get_attribute(index, column="dac_token")
        
        t5_embedding = torch.from_numpy(np.frombuffer(t5_feat, dtype=np.float32).reshape(-1, self.t5_dim))
        clip_embedding = torch.from_numpy(np.frombuffer(clip_feat, dtype=np.float32).reshape(-1, self.clip_dim))
        audio_tokens = torch.from_numpy(np.frombuffer(dac_token, dtype=np.int16)).long()
        if self.double_codebook:
            audio_tokens = audio_tokens.reshape(2, -1)
            if not self.use_first_code_book:
                if audio_tokens.shape[1] > self.audio_token_max_length:
                    audio_tokens = audio_tokens[:, :self.audio_token_max_length]
                audio_tokens[0,:] += self.audio_token_offset
                audio_tokens[1,:] += (self.audio_token_offset + self.dac_voca_size)
                if self.codebook_arrangement == "interleave":
                    # flatten audio token in a intelevaved manner
                    audio_tokens = audio_tokens.permute(1, 0).reshape(1, -1).contiguous().squeeze(0)
                else:
                    audio_tokens = audio_tokens.flatten()
            else:
                audio_tokens = audio_tokens[0] + self.audio_token_offset
        else:
            audio_tokens += + self.audio_token_offset
        # change to drop together
        drop_clip = random.random() < self.drop_clip
        drop_text = random.random() < self.drop_text
        if self.double_codebook and not self.use_first_code_book:
            max_auido_token_len = self.audio_token_max_length * 2 
        else:
            max_auido_token_len = self.audio_token_max_length
        if self.use_t5:
            input_tokens = self.tokenizer.encode_visual_t5_audio_sequence(
                audio_tokens,
                max_len=self.max_sequence_len,
                max_audio_len=max_auido_token_len,
                max_clip_frame=self.max_frames,
                max_t5_len=self.max_text_length
            )
        else:
            if self.no_text:
                input_tokens = self.tokenizer.encode_visual_audio_sequence_no_text(
                    audio_tokens, 
                    max_len=self.max_sequence_len, 
                    max_audio_len=max_auido_token_len, 
                    max_clip_frame=self.max_frames
                )
            else:
                structure_caption = self.index_manager.get_attribute(index, column="structure_caption")
                input_tokens = self.tokenizer.encode_visual_audio_sequence(audio_tokens, structure_caption, drop_text)
        target_token = input_tokens.clone()
        # TODO: make it configurable
        if self.use_t5:
            offset = max_auido_token_len + 4     # audio token and 4 special token
            target_token[:-offset] = -100      # visual and T5 part.
        else:
            clip_start = 2
            clip_end = clip_start + self.max_frames
            target_token[clip_start:clip_end] = -100       # only visual part
        target_token[target_token == self.tokenizer.special_token_map["<pad>"]] = -100

        if clip_embedding.shape[0] > self.max_frames:
            clip_embedding = clip_embedding[:self.max_frames]
        padeed_clip_embedding = self.clip_pad_emb.clone()
        if not drop_clip:
            padeed_clip_embedding[:clip_embedding.shape[0]] = clip_embedding

        if t5_embedding.shape[0] > self.max_text_length:
            t5_embedding = t5_embedding[:self.max_text_length]
        padded_t5_embedding = self.t5_pad_emb.clone()
        if not drop_text:
            padded_t5_embedding[:t5_embedding.shape[0]] = t5_embedding

        data_dict = {
            "clip_embedding": padeed_clip_embedding, 
            "t5_embedding": padded_t5_embedding, 
            "tokens": input_tokens, 
            "target": target_token,
        }
        return data_dict

    def shuffle(self, seed, fast=False):
        self.index_manager.shuffle(seed, fast=fast)

    def __len__(self):
        if self.debug:
            return min(len(self.index_manager), 4096)
        return len(self.index_manager)


class VisualAudioIndexMultiHeadDataset(VisualAudioIndexDataset):
    def __init__(self, audio_pad_token, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_pad_token = audio_pad_token
    
    def pad_audio(self, audio_tokens, max_len=0):
        if audio_tokens.shape[1] < max_len:
            padd_token = torch.ones(audio_tokens.shape[0], max_len - audio_tokens.shape[1]).long() * self.audio_pad_token
            audio_tokens = torch.cat([audio_tokens, padd_token], dim=1)
        elif audio_tokens.shape[1] > max_len:
            audio_tokens = audio_tokens[:, :max_len]
        return audio_tokens

    def __getitem__(self, index):
        if self.feat_in_shadow:
            clip_feat, t5_feat, dac_token = self.get_data_from_shadow(index)
        else:
            clip_feat = self.index_manager.get_attribute(index, column="clip_feat")
            t5_feat = self.index_manager.get_attribute(index, column="t5_feat")
            dac_token = self.index_manager.get_attribute(index, column="dac_token")
              
        t5_embedding = torch.from_numpy(np.frombuffer(t5_feat, dtype=np.float32).reshape(-1, self.t5_dim))
        clip_embedding = torch.from_numpy(np.frombuffer(clip_feat, dtype=np.float32).reshape(-1, self.clip_dim))
        audio_tokens = torch.from_numpy(np.frombuffer(dac_token, dtype=np.int64)).long()
        audio_tokens = audio_tokens.reshape(2, -1)
        max_auido_token_len = self.audio_token_max_length
        audio_tokens = self.pad_audio(audio_tokens, max_len=max_auido_token_len)
        target_token = audio_tokens.clone()
        target_token[target_token == self.audio_pad_token] = -100

        # form input token for transformer
        drop_text = random.random() < self.drop_text
        input_tokens = self.tokenizer.encode_visual_t5_audio_placeholder(
            max_len=self.max_sequence_len,
            max_audio_len=max_auido_token_len,
            max_clip_frame=self.max_frames,
            max_t5_len=self.max_text_length
        )

        if clip_embedding.shape[0] > self.max_frames:
            clip_embedding = clip_embedding[:self.max_frames]
        padeed_clip_embedding = self.clip_pad_emb.clone()
        if random.random() >= self.drop_clip:
            padeed_clip_embedding[:clip_embedding.shape[0]] = clip_embedding

        if t5_embedding.shape[0] > self.max_text_length:
            t5_embedding = t5_embedding[:self.max_text_length]
        padded_t5_embedding = self.t5_pad_emb.clone()
        if not drop_text:
            padded_t5_embedding[:t5_embedding.shape[0]] = t5_embedding

        data_dict = {
            "clip_embedding": padeed_clip_embedding, 
            "t5_embedding": padded_t5_embedding, 
            "tokens": input_tokens, 
            "target": target_token,
            "audio_tokens": audio_tokens,
        }
        return data_dict
    

def build_visual_audio_dataloader(cfg, rank, world_size):
    use_multi_head = cfg.get("use_multi_head", False)
    if use_multi_head:
        audio_pad_token = cfg.audio_token_max_length
        dataset_cls = partial(VisualAudioIndexMultiHeadDataset, audio_pad_token=audio_pad_token)
    else:
        dataset_cls = VisualAudioIndexDataset
    dataset = dataset_cls(
        index_file=cfg.index_file,
        text_tokenizer=cfg.text_tokenizer,
        clip_pad_emb=cfg.clip_pad_emb,
        t5_pad_emb=cfg.t5_pad_emb,
        use_t5=cfg.use_t5,
        audio_token_offset=cfg.audio_token_offset,
        drop_clip=cfg.drop_clip,
        drop_text=cfg.drop_text,
        index_kwargs=dict(
            batch_size=cfg.micro_batch_size,
            world_size=world_size,
            **cfg.get("index_kwargs", {}),
        ),
        debug=False,
        shadow_file_cfg=cfg.get("shadow_file", {}),
        feat_in_shadow=cfg.get("feat_in_shadow", False),
        double_codebook=cfg.get("double_codebook", False),
        dac_voca_size=cfg.get("dac_voca_size", 4096),
        max_sequence_len=cfg.get("max_sequence_len", 1025),
        max_frames=cfg.max_frames,
        max_text_length=cfg.max_text_length,
        codebook_arrangement=cfg.get("codebook_arrangement", "interleave"),
        use_first_code_book=cfg.get("use_first_code_book", False),
        no_text=cfg.get("no_text", False),
        use_custom_tokenizer=cfg.get("use_custom_tokenizer", False),
    )
    sampler = DistributedSamplerWithStartIndex(
        dataset, 
        shuffle=False, 
        num_replicas=world_size, 
        rank=rank, 
        seed=cfg.global_seed, 
        drop_last=True
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_params["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        worker_init_fn=set_worker_seed_builder(rank)
    )
    return dataloader, dataset, sampler
