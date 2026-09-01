import random

import nltk
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from index_kits import ArrowIndexV2, MultiIndexV2, arrow_mapper
from index_kits.sampler import DistributedSamplerWithStartIndex
from torch.utils.data import Dataset, DataLoader

from hymm.utils.torch_utils import set_worker_seed_builder


NLTK_PRE_DATA_PATH = "/apdcephfs_gy2/share_302507476/yutaocui/model_zoo/nltk_data"
MEANINGLESS_WORDS = ["is", "are", "being", "am", "be", "was", "were", "with", "object", "objects", "something"]


class VisualAudioDiffusionFixedLenIndexDataset(Dataset):
    def __init__(
        self,
        index_file,
        sync_pad_emb=None,
        index_kwargs=None,
        logger=None,
        clip_dim=768,
        max_clip_length=64,  # 8s-8fps
        max_text_length=181,
        audio_vae_latent_max_length=400,  # 8s
        max_sync_length=192,  # 8s
        debug=False,
        drop_visual=0.0,
        drop_text=0.0,
        visualdata_drop_text=None,
        feat_in_shadow=False,
        shadow_file_cfg={},
        audio_vae_latent_dim=64,
        sync_modulation=True,
        sync_feat_dim=768,
        use_clip_text_feat=False,
        keywords_training_ratio=0,
        enable_quality_tag_for_pureauio=False,
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

        self.use_clip_text_feat = use_clip_text_feat
        self.sync_modulation = sync_modulation
        self.sync_feat_dim = sync_feat_dim
        self.max_sync_length = max_sync_length
        self.feat_in_shadow = feat_in_shadow
        # 针对视频-音频数据单独进行drop_text
        self.visualdata_drop_text = visualdata_drop_text
        # sft时用keywords来代替完整caption进行训练的比例，默认为0
        self.keywords_training_ratio = keywords_training_ratio
        self.enable_quality_tag_for_pureauio = enable_quality_tag_for_pureauio
        self.drop_visual = drop_visual
        self.drop_text = drop_text
        self.clip_dim = clip_dim
        self.max_clip_length = max_clip_length
        self.audio_vae_latent_max_length = audio_vae_latent_max_length
        self.max_text_length = max_text_length
        self.audio_vae_latent_dim = audio_vae_latent_dim
        self.clip_pad_emb = torch.zeros((max_clip_length, self.clip_dim), dtype=torch.float32)

        self.debug = debug

        if self.keywords_training_ratio > 0:
            nltk.data.path.append(NLTK_PRE_DATA_PATH)

        shadow_file_fn = {}
        if feat_in_shadow:
            self.logger.info(shadow_file_cfg)
            shadow_file_fn.update({"clip": partial(arrow_mapper, suffix=shadow_file_cfg["clip_suffix"])})
            shadow_file_fn.update({"vae": partial(arrow_mapper, suffix=shadow_file_cfg["vae_suffix"])})
            self.clip_col = shadow_file_cfg["clip_col"]
            # self.t5_col = shadow_file_cfg["t5_col"]
            self.vae_col = shadow_file_cfg["vae_col"]

            if self.sync_modulation:
                shadow_file_fn.update({"sync": partial(arrow_mapper, suffix=shadow_file_cfg["sync_suffix"])})
                self.sync_col = shadow_file_cfg["sync_col"]
                self.sync_pad_emb = torch.load(sync_pad_emb, "cpu")[:max_sync_length]
                assert list(self.sync_pad_emb.shape) == [max_sync_length, sync_feat_dim]

            if self.use_clip_text_feat:
                shadow_file_fn.update({"caption": partial(arrow_mapper, suffix="_audiosetcaps")})
                self.caption_col = "SoundCaption"

        index_load_kwargs["shadow_file_fn"] = shadow_file_fn
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"Using {self.index_manager}")

    def extract_nouns(self, caption, meaningless_words=MEANINGLESS_WORDS):
        words = nltk.word_tokenize(caption)
        pos_tags = nltk.pos_tag(words)
        nouns = [word for word, pos in pos_tags if (pos.startswith("NN") and word not in meaningless_words)]
        return nouns

    def extract_verbs(self, caption, meaningless_words=MEANINGLESS_WORDS):
        words = nltk.word_tokenize(caption)
        pos_tags = nltk.pos_tag(words)
        verbs = [word for word, pos in pos_tags if pos.startswith("VB") and word not in meaningless_words]
        return verbs

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
        caption = ""
        if self.use_clip_text_feat:
            caption = self.index_manager.get_attribute(index, column=self.caption_col, shadow="caption")
            t5_feat = None
        else:
            raise NotImplementedError
            # t5_feat = self.index_manager.get_attribute(index, column=self.t5_col, shadow="t5")

        clip_feat = self.index_manager.get_attribute(index, column=self.clip_col, shadow="clip")
        sync_feat = None
        if self.sync_modulation:
            sync_feat = self.index_manager.get_attribute(index, column=self.sync_col, shadow="sync")

        vae_mean = self.index_manager.get_attribute(index, column="audio_vae_mean", shadow="vae")
        vae_std = self.index_manager.get_attribute(index, column="audio_vae_std", shadow="vae")
        return caption, clip_feat, t5_feat, sync_feat, vae_mean, vae_std

    def __getitem__(self, index):
        # ======================================= Get feature from arrow =======================================
        if self.feat_in_shadow:
            caption, clip_feat, t5_feat, sync_feat, vae_mean, vae_std = self.get_data_from_shadow(index)

        # 特殊处理 caption
        if self.keywords_training_ratio > 0 and random.random() < self.keywords_training_ratio:
            nouns = self.extract_nouns(caption)
            verbs = self.extract_verbs(caption)
            unique_words = set(nouns) | set(verbs)
            caption = ", ".join(unique_words)
        if self.enable_quality_tag_for_pureauio and sync_feat is None and clip_feat is None:
            # FIXME (yutaocui): 当caption的末尾是'.'时要先去掉
            caption = caption + ", high-quality"

        # 8s synchformer feature
        if sync_feat is not None:
            sync_embedding = torch.from_numpy(
                np.frombuffer(sync_feat, dtype=np.float32).reshape(-1, self.sync_feat_dim)
            )
            if sync_embedding.shape[0] >= self.max_sync_length:
                sync_embedding = sync_embedding[: self.max_sync_length]
            else:
                sync_embedding = None
        else:
            sync_embedding = None
        padded_sync_embedding = self.sync_pad_emb[: self.max_sync_length].clone()

        # 8s vae feature
        assert vae_mean is not None, f"vae latents must be prepared, not be None"
        vae_mean = torch.from_numpy(np.frombuffer(vae_mean, dtype=np.float32))
        vae_mean = vae_mean.reshape(self.audio_vae_latent_dim, -1)
        # FIXME: 用dac-vae训练不能做norm，不然loss巨大，需要找一下原因
        audio_tokens = vae_mean

        if audio_tokens.shape[1] >= self.audio_vae_latent_max_length:
            audio_tokens = audio_tokens[:, : self.audio_vae_latent_max_length]
        elif self.audio_vae_latent_max_length - audio_tokens.shape[1] <= 5:
            # 处理一些略微小于8s的视频
            pad_audio_tokens = audio_tokens[:, -1:].repeat(1, self.audio_vae_latent_max_length - audio_tokens.shape[1])
            audio_tokens = torch.cat((audio_tokens, pad_audio_tokens), dim=-1)
        assert (
            audio_tokens.shape[1] == self.audio_vae_latent_max_length
        ), f"audio_tokens len is {audio_tokens.shape[1]}, which less than the expected {self.audio_vae_latent_max_length}"

        # 8s clip feature
        if clip_feat is not None:
            clip_embedding = torch.from_numpy(np.frombuffer(clip_feat, dtype=np.float32).reshape(-1, self.clip_dim))
            if clip_embedding.shape[0] >= self.max_clip_length:
                clip_embedding = clip_embedding[: self.max_clip_length]
            else:
                clip_embedding = None
        else:
            clip_embedding = None
        padded_clip_embedding = self.clip_pad_emb[: self.max_clip_length].clone()

        # =======================================  Drop text or visual feature for unconditional training =======================================
        drop_text = random.random() < self.drop_text
        drop_visual = random.random() < self.drop_visual or clip_embedding is None or sync_embedding is None
        # 针对视频数据单独进行drop_text, 设置较大的比例
        visual_data = clip_embedding is not None and sync_embedding is not None
        if visual_data and self.visualdata_drop_text is not None:
            drop_text = random.random() < self.visualdata_drop_text

        padded_caption = ""
        if not drop_text:
            padded_caption = caption
        if not drop_visual:
            padded_clip_embedding = clip_embedding
            padded_sync_embedding = sync_embedding

        data_dict = {
            "clip_embedding": padded_clip_embedding,
            # "t5_embedding": padded_t5_embedding,  # 最终版本没用到注释了
            "caption": padded_caption,
            "audio_token": audio_tokens,
            "drop_text": drop_text,
            "drop_visual": drop_visual,
            "sync_embedding": padded_sync_embedding,
        }
        return data_dict

    def shuffle(self, seed, fast=False):
        self.index_manager.shuffle(seed, fast=fast)

    def __len__(self):
        if self.debug:
            return min(len(self.index_manager), 4096)
        return len(self.index_manager)


def build_visual_audio_diffusion_fixedlen_dataloader(cfg, rank, world_size):
    dataset = VisualAudioDiffusionFixedLenIndexDataset(
        index_file=cfg.index_file,
        sync_pad_emb=cfg.get("sync_pad_emb", None),
        drop_visual=cfg.drop_clip,
        drop_text=cfg.drop_text,
        clip_dim=cfg.clip_dim,
        sync_modulation=cfg.get("sync_modulation", False) or cfg.get("add_sync_feat_to_audio", False),
        sync_feat_dim=cfg.get("sync_feat_dim", 768),
        index_kwargs=dict(
            batch_size=cfg.micro_batch_size,
            world_size=world_size,
            **cfg.get("index_kwargs", {}),
        ),
        shadow_file_cfg=cfg.get("shadow_file", {}),
        feat_in_shadow=cfg.get("feat_in_shadow", False),
        audio_vae_latent_max_length=cfg.get("audio_vae_latent_max_length", 400),
        max_clip_length=cfg.get("max_clip_length", 64),
        max_text_length=cfg.max_text_length,
        max_sync_length=cfg.get("max_sync_length", 192),
        audio_vae_latent_dim=cfg.get("audio_vae_latent_dim", 64),
        use_clip_text_feat=cfg.get("use_clip_text_feat", False),
        visualdata_drop_text=cfg.get("visualdata_drop_text", None),
        keywords_training_ratio=cfg.get("keywords_training_ratio", 0),
        enable_quality_tag_for_pureauio=cfg.get("enable_quality_tag_for_pureauio", False),
        debug=False,
    )
    sampler = DistributedSamplerWithStartIndex(
        dataset, shuffle=False, num_replicas=world_size, rank=rank, seed=cfg.global_seed, drop_last=True
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_params["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        worker_init_fn=set_worker_seed_builder(rank),
    )
    return dataloader, dataset, sampler
