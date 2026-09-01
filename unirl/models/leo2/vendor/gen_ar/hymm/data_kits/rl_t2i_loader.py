import random
import time

import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass, field, make_dataclass, asdict

from .index_dataset import IndexColumn, resample_on_gray
from ..data_kits.instruction_template import (
    text2image_instructions,
)
from .t2i_loader import TextImageArrowStream, TextImageData


class RLTextImageArrowStream(TextImageArrowStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_sample_win_lose_ratio = kwargs['args'].get("random_sample_win_lose_ratio", 0.0)
        self.win_image_col_prefix = kwargs['args'].get("win_image_col_prefix", "win_cache_image_")
        self.lose_image_col_prefix = kwargs['args'].get("lose_image_col_prefix", "lose_cache_image_")
        self.win_image_col_2 = kwargs['args'].get("win_image_col_2", "cache_image")
        self.use_sft_loss_col = kwargs['args'].get("use_sft_loss_col", None)

    def parse_columns_and_register_shadow(self, index_kwargs):
        self.win_imgs_num_col = IndexColumn(index_kwargs.get("win_imgs_num_col"), self, self.logger)
        self.lose_imgs_num_col = IndexColumn(index_kwargs.get("lose_imgs_num_col"), self, self.logger)
        self.image_text_col = IndexColumn(index_kwargs.get("image_text_col"), self, self.logger)
        self.image_caption_col = IndexColumn(index_kwargs.get("image_caption_col"), self, self.logger)
        self.image_caption_col_2 = IndexColumn(index_kwargs.get("image_caption_col_2"), self, self.logger)
        self.image_caption_col_3 = IndexColumn(index_kwargs.get("image_caption_col_3"), self, self.logger)
        self.clip_score_col = IndexColumn(index_kwargs.get("clip_score_col"), self, self.logger)
        self.seed_col = IndexColumn(index_kwargs.get("seed_col"), self, self.logger)
        self.subtask_col = IndexColumn(index_kwargs.get("subtask_col"), self, self.logger)

        # Task related settings for getting data
        if "src_image_col" in index_kwargs:
            self.src_image_col = IndexColumn(index_kwargs.get("src_image_col"), self, self.logger)
        else:
            self.src_image_col = None

        if "ref_image_col" in index_kwargs:
            self.ref_image_col = IndexColumn(index_kwargs.get("ref_image_col"), self, self.logger)
        else:
            self.ref_image_col = None

        if "use_face_reward_col" in index_kwargs:
            self.use_face_reward_col = IndexColumn(index_kwargs.get("use_face_reward_col"), self, self.logger)
        else:
            self.use_face_reward_col = None

        if "sem_points_col" in index_kwargs:
            self.sem_points_col = IndexColumn(index_kwargs.get("sem_points_col"), self, self.logger)
        else:
            self.sem_points_col = None
                
        if "subtask_col" in index_kwargs:
            self.subtask_col = IndexColumn(index_kwargs.get("subtask_col"), self, self.logger)
        else:
            self.subtask_col = None

        self.short_caption_col = index_kwargs.get("short_caption_col", None)
        self.no_load_image = index_kwargs.get("no_load_image", False)
        self.use_reward_tags = index_kwargs.get("use_reward_tags", False)
        self.task_type = index_kwargs.get("task_type", "grpo")

        # Register other columns. Do not remove the previous 
        self.register_caption_cols(index_kwargs)
        self.register_extra_cols(index_kwargs)
    
    def register_caption_cols(self, index_kwargs):
        # 自动注册 caption 的 columns
        for key, value in index_kwargs.items():
            if "col" in key and "image_caption" in key:
                # 确保 key 不是 self 的属性
                if hasattr(self, key):
                    continue
                setattr(self, key, IndexColumn(value, self, self.logger))

    def register_extra_cols(self, index_kwargs):
        # 自动注册额外的 columns
        for key, value in index_kwargs.items():
            if "col" in key and "extra" in key:
                # 确保 key 不是 self 的属性
                if hasattr(self, key):
                    continue
                setattr(self, key, IndexColumn(value, self, self.logger))

    def get_use_face_reward(self, index, **use_face_reward_col):
        try:
            use_face_reward = self.index_manager.get_attribute(index, **use_face_reward_col)
            if isinstance(use_face_reward, str):
                use_face_reward = use_face_reward.lower() == "true"
            elif isinstance(use_face_reward, bool):
                pass
            else:
                raise ValueError(f"Unsupported use_face_reward type: {type(use_face_reward)}")
        except Exception as e:
            self.logger.error(f"({use_face_reward_col=}, {index=}) {type(e)}: {e}. Fallback to False.")
            use_face_reward = False
        return use_face_reward

    def get_sem_points(self, index, **sem_points_col):
        try:
            sem_points = self.index_manager.get_attribute(index, **sem_points_col)
        except Exception as e:
            self.logger.error(f"({sem_points_col=}, {index=}) {type(e)}: {e}. Fallback to empty list.")
            sem_points = ""
        return sem_points

    def get_subtask(self, index, **subtask_col):
        try:
            subtask = self.index_manager.get_attribute(index, **subtask_col)
        except Exception as e:
            self.logger.error(f"({subtask_col=}, {index=}) {type(e)}: {e}. Fallback to empty string.")
            subtask = ""
        return subtask
    
    def get_raw_image(self, index, **image_col):
        start = time.time()
        sleep_time = 1
        while True:
            try:
                path = self.index_manager.get_attribute(index, **image_col)
                if "url" in image_col['column'] or path.startswith("http"):
                    image = self.get_image_from_url_cos(index, **image_col)
                elif 'cache_image' in image_col['column']:
                    if isinstance(index, (int, np.integer)):
                        image_path = self.index_manager.get_attribute(index, **image_col)
                    else:
                        image_path = index
                    image_path = self.parse_image_path(image_path)
                    image = Image.open(image_path).convert("RGB")
                else:
                    # bytes image
                    image = self.get_image_from_arrow(index, **image_col)
                image_flag = "normal"
                break
            except Exception as e:
                self.logger.error(f"({image_col=}, {index=}) {type(e)}: {e}. Fallback to gray image.")
                image = Image.new("RGB", (self.training_image_size[0], self.training_image_size[1]), (128, 128, 128))
                image_flag = "gray"
                break
        if sleep_time > 1:
            self.logger.info(f"({image_col=}, {index=}) Loaded image after retrying {time.time() - start} seconds.")
        return image, image_flag

    @resample_on_gray
    def get_dpo_t2i_data(self, index):
        # Get win-image and lose-image
        try:
            lose_imgs_num = int(self.index_manager.get_attribute(index, **self.lose_imgs_num_col))
        except Exception as e:
            lose_imgs_num = 0

        if random.random() < self.random_sample_win_lose_ratio:
            try:
                win_imgs_num = int(self.index_manager.get_attribute(index, **self.win_imgs_num_col))
                lose_imgs_num = int(self.index_manager.get_attribute(index, **self.lose_imgs_num_col))
            except:
                win_imgs_num = 1
                lose_imgs_num = 1
            win_img_id = random.randint(0, win_imgs_num - 1)
            lose_img_id = random.randint(0, lose_imgs_num - 1)
        else:
            win_img_id = 0
            lose_img_id = 0
        win_image_col = {"column": self.win_image_col_prefix + str(win_img_id)}
        lose_image_col = {"column": self.lose_image_col_prefix + str(lose_img_id)}
        # `random_crop` is recommended to set to False, since to ensure the consistency of the win-image and lose-image
        columns = self.index_manager.get_columns(index)
        if win_image_col['column'] in columns:
            win_image, _, image_flag = self.get_image_with_size(index, random_crop=False, **win_image_col)
        elif self.win_image_col_2 in columns:
            win_image_col2 = {"column": self.win_image_col_2}
            win_image, _, image_flag = self.get_image_with_size(index, random_crop=False, **win_image_col2)
        else:
            raise ValueError(f"Win image column {win_image_col['column']} or {self.win_image_col_2} is not found in the index columns.")
        
        if lose_imgs_num == 0:
            lose_image = win_image
            lose_exist = False
        else:
            lose_image, _, _ = self.get_image_with_size(index, random_crop=False, **lose_image_col)
            lose_exist = True

        # Get text
        prompt = self.get_text(index)
        # -- for cot
        if isinstance(prompt, tuple):
            prompt, recaption = prompt
        else:
            recaption = None
        if self.template == "instruct":
            instruction = self.get_system_prompt(text2image_instructions, with_space=True)
            prompt = f"{instruction}{prompt}"

        return lose_image, lose_exist, TextImageData(
            prompt=prompt, tgt_image=win_image,
            recaption=recaption,
            image_flag=image_flag,
            index=index,
        )

    def _getitem_dpo(self, index):
        dtype = 't2i'
        lose_image, lose_exist, data = self.get_dpo_t2i_data(index)
        index = data.index

        # Get use_sft_loss if determined
        if self.use_sft_loss_col is not None:
            use_sft_loss = self.index_manager.get_attribute(index, **self.use_sft_loss_col)
            use_sft_loss = use_sft_loss == "true" or use_sft_loss == True or int(use_sft_loss) == 1
        else:
            use_sft_loss = False

        # Get seed
        try:
            seed = self.index_manager.get_attribute(index, 'seed')
        except Exception as e:
            seed = np.random.randint(0, 1_000_000)
        
        # 1 is <bos>.
        # <eos> is not included because it will be stripped in the shift of next-token-prediction
        extra_num_tokens = 1 + data.num_special_tokens + self.dummy_number
        # uncondition
        do_uncond = (self.uncond_p > 0) and (random.random() < self.uncond_p)
        uncond_kwargs = dict(uncond_enabled=do_uncond, uncond_p=(1.0 if do_uncond else 0.0))
        # We want the image prefix special tokens <boi>, <img_size_*>, and <img_ratio_*> to be learned.
        # It is implemented by adding text mask end offsets to the prefix text sections.
        num_image_prefix = 1 + (2 if self.add_image_shape_token else 0)

        if self.template == "pretrain":
            # xxxx<boi>[image]<eoi>
            sections = [
                dict(type="text", text=data.prompt, max_length=self.text_token_length - extra_num_tokens,
                     **uncond_kwargs, ignore=do_uncond or self.ignore_text_ntp),
                dict(type="text", text='', ignore=False, end_offset=num_image_prefix),
                dict(type="gen_image", **data.tgt_image.i.meta_info),
                dict(type="text", text='', ignore=False, end_offset=1)    # include <eos> token, 1 token
            ]

        elif self.template == "instruct":
            # User: xxxx\n\nAssistant: <answer><boi>[image]<eoi></answer>
            extra_num_tokens += 9  # "User: " + "\n\n" + "Assistant: <answer>" + </answer>

            user_prefix_section = [
                dict(type="text", text=f"{self.roles[0]}: ", ignore=True), # "User: " 3 tokens
            ]
            prompt_and_bot_prefix_section = [
                dict(type="text", text=data.prompt, max_length=self.text_token_length - extra_num_tokens,
                     **uncond_kwargs, ignore=True),
                dict(type="text", text=self.default_conv.sep, ignore=True), # "\n\n" 1 token
                dict(type="text", text=f"{self.roles[1]}: <answer>", ignore=True), # "Assistant: <answer>" 4 tokens
            ]
            gen_section = [
                dict(type="text", text='', ignore=True if do_uncond else False, end_offset=num_image_prefix),
                dict(type="gen_image", **data.tgt_image.i.meta_info),
                dict(type="text", text="</answer>", ignore=False),
                dict(type="text", text='', ignore=False, end_offset=1)
            ]
            if data.recaption:  # if not None and not empty
                cot_section = [
                    dict(type="text", text="<recaption>", ignore=True),
                    dict(type="text", text=data.recaption, ignore=do_uncond,
                         max_length=self.description_token_length - 2, **uncond_kwargs),
                    dict(type="text", text="</recaption>", ignore=do_uncond),
                ]
            else:
                cot_section = []

            sections = user_prefix_section + prompt_and_bot_prefix_section + cot_section + gen_section
        else:
            raise ValueError(f"Unsupported template: {self.template}")

        if self.sequence_pack:
            max_token_length = None
            add_pad = False
        else:
            max_token_length = (self.text_token_length + self.image_token_length + 1 - self.dummy_number +
                                self.description_token_length)
            add_pad = 'auto'
        # Build template and encode tokens
        output = self.tokenizer.encode_general(
            sections=sections,
            max_token_length=max_token_length,
            add_pad=add_pad,
        )
        target_token = output.tokens.clone()
        target_token[output.text_mask == 0.0] = -100

        # Prepare attention mask
        if self.task_kwargs.get('attn_type', 'auto') == 'auto':
            n_tokens = output.tokens.shape[0] - int(self.attn_mask_seq_m1) + self.dummy_number
            attention_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
            for image_slice in output.gen_image_slices:
                attention_mask[image_slice, image_slice] = True
            attention_mask = attention_mask.unsqueeze(0)    # head dim
        else:
            attention_mask = None

        # 2d rope
        rope_image_info = self.get_rope_image_info(sections, output)

        ret = {
            "task_type": self.task_type,
            "data_type": "t2i",
            "dtype": dtype,
            "seed": seed,
            "text": data.prompt,
            "image": data.tgt_image,                # (3, H~, W~)
            "lose_image": lose_image,               # (3, H~, W~)
            "lose_exist": lose_exist,
            "n_samples": 1,                         # ()
            "tokens": output.tokens,                # (L), L = text_token_length + 1 + image_token_length
            "target_tokens": target_token,          # (L)
            "text_mask": output.text_mask,          # (L)
            "image_mask": output.gen_image_mask,    # (L)
        }
        if attention_mask is not None:
            ret["attention_mask"] = attention_mask
        else:
            ret["gen_image_slices"] = output.gen_image_slices
        if output.iw_ih_scatter_index is not None:
            ret.update({
                "iw_ih_scatter_index": output.iw_ih_scatter_index,  # (2)
                "iw_ih_scatter_src": data.iw_ih_scatter_src,        # (2)
            })
        if output.timestep_scatter_index is not None:
            ret.update({
                "timestep_scatter_index": output.timestep_scatter_index,   # (1)
            })
        if rope_image_info is not None:
            ret.update({
                "rope_image_info": rope_image_info,  # (2)
            })
        if self.use_sft_loss_col is not None:
            ret["use_sft_loss"] = use_sft_loss

        return ret

    @resample_on_gray
    def _getitem_grpo(self, index):
        # Get text
        text = self.get_text(index)
        # Get seed
        try:
            seed = self.index_manager.get_attribute(index, 'seed')
        except Exception as e:
            seed = np.random.randint(0, 1_000_000)
        
        if self.use_reward_tags:
            reward_tags = self.index_manager.get_attribute(index, 'reward_tags')
            if reward_tags is None:
                raise ValueError("reward_tags is not found in the index.")

        # TODO: this should be renamed
        dtype = self.index_manager.get_attribute(index, 'task')  # t2i or ti2i
        assert dtype in ['t2i', 'ti2i'], dtype
        if dtype == 'ti2i':
            src_image, src_image_flag = self.get_raw_image(index, **self.src_image_col)  # pil_image
        else:
            src_image = src_image_flag = None

        if dtype == 'ti2i' and self.ref_image_col is not None:
            ref_image, ref_image_flag = self.get_raw_image(index, **self.ref_image_col)  # pil_image
        else:
            ref_image = ref_image_flag = None

        if self.use_face_reward_col is not None:
            use_face_reward = self.get_use_face_reward(index, **self.use_face_reward_col)
        else:
            use_face_reward = False

        if self.sem_points_col is not None:
            sem_points = self.get_sem_points(index, **self.sem_points_col)
        else:
            sem_points = ""

        if self.subtask_col is not None:
            subtask = self.get_subtask(index, **self.subtask_col)
        else:
            subtask = ""

        ret = {
            "task_type": self.task_type,
            "data_type": dtype,
            "dtype": dtype,
            "seed": seed,
            "text": text,
            "n_samples": 1,
        }
        if self.use_reward_tags:
            ret["reward_tags"] = reward_tags
        if src_image is not None:
            ret["src_image"] = src_image
        if ref_image is not None:
            ret["ref_image"] = ref_image
        if self.short_caption_col is not None:
            short_caption = self.index_manager.get_attribute(index, self.short_caption_col)
            ret["short_caption"] = short_caption
        if use_face_reward is not None:
            ret["use_face_reward"] = use_face_reward
        if sem_points is not None:
            ret["sem_points"] = sem_points
        if subtask is not None:
            ret["subtask"] = subtask

        if dtype == 'ti2i' and (src_image_flag == 'gray' or ref_image_flag == 'gray'):
            ret["image_flag"] = 'gray'
        else:
            ret["image_flag"] = 'normal'
        ret = make_dataclass('RLItem', ret.items())(**ret)
        return ret

    def __getitem__(self, index):
        if "dpo" in self.task_type:
            return self._getitem_dpo(index)
        elif "grpo" in self.task_type:
            return asdict(self._getitem_grpo(index))
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    @staticmethod
    def collate_fn(batch):
        data_type = [item["data_type"] for item in batch]
        dtype = [item["dtype"] for item in batch]
        text = [item["text"] for item in batch]
        seeds = [item["seed"] for item in batch]
        n_samples = torch.tensor([item["n_samples"] for item in batch])
        
        if "reward_tags" in batch[0]:
            reward_tags = [item["reward_tags"] for item in batch]
        else:
            reward_tags = None

        if "src_image" in batch[0]:
            src_images = [item["src_image"] for item in batch]
        else:
            src_images = None

        if "ref_image" in batch[0]:
            ref_images = [item["ref_image"] for item in batch]
        else:
            ref_images = None

        if "use_face_reward" in batch[0]:
            use_face_rewards = [item["use_face_reward"] for item in batch]
        else:
            use_face_rewards = None
        if "sem_points" in batch[0]:
            sem_points = [item["sem_points"] for item in batch]
        else:
            sem_points = None
        if "subtask" in batch[0]:
            subtask = [item["subtask"] for item in batch]
        else:
            subtask = None

        if "grpo" in batch[0]["task_type"]:
            ret = {
                "data_type": data_type,
                "dtype": dtype,
                "text": text,
                "seeds": seeds,
                "n_samples": n_samples,
                "src_images": src_images,
                "ref_images": ref_images,
                "use_face_rewards": use_face_rewards,
                "sem_points": sem_points,
                "subtask": subtask,
            }
            if reward_tags is not None:
                ret["reward_tags"] = reward_tags
            if "short_caption" in batch[0]:
                short_caption = [item["short_caption"] for item in batch]
                ret["short_caption"] = short_caption

            # ret = {key: value for key, value in ret.items() if value is not None}
            return ret

        image = [item["image"] for item in batch]
        can_stack = isinstance(image, torch.Tensor) or src_images is None
        if all(isinstance(x, torch.Tensor) for x in image) and all(x.shape == image[0].shape for x in image):
            image = torch.stack(image)
        else:
            can_stack = False
        
        # lose_image for dpo
        if "lose_image" in batch[0]:
            lose_image = [item["lose_image"] for item in batch]
            if all(isinstance(x, torch.Tensor) for x in lose_image) and all(x.shape == lose_image[0].shape for x in lose_image):
                lose_image = torch.stack(lose_image)
        else:
            lose_image = None
        if "lose_exist" in batch[0]:
            lose_exist = [item["lose_exist"] for item in batch]
        else:
            lose_exist = None

        tokens = torch.stack([item["tokens"] for item in batch])
        target_tokens = torch.stack([item["target_tokens"] for item in batch])
        text_mask = torch.stack([item["text_mask"] for item in batch])
        image_mask = torch.stack([item["image_mask"] for item in batch])

        if "iw_ih_scatter_index" in batch[0]:
            iw_ih_scatter_index = [item["iw_ih_scatter_index"] for item in batch]
            iw_ih_scatter_src = [item["iw_ih_scatter_src"] for item in batch]
            if can_stack:
                iw_ih_scatter_index = torch.stack(iw_ih_scatter_index)
                iw_ih_scatter_src = torch.stack(iw_ih_scatter_src)
        else:
            iw_ih_scatter_index = None
            iw_ih_scatter_src = None

        if "timestep_scatter_index" in batch[0]:
            timestep_scatter_index = [item["timestep_scatter_index"] for item in batch]
            if can_stack:
                timestep_scatter_index = torch.stack(timestep_scatter_index)
        else:
            timestep_scatter_index = None

        attention_mask = torch.stack([item["attention_mask"] for item in batch]) if "attention_mask" in batch[0] else None

        gen_image_slices = [item["gen_image_slices"] for item in batch] if "gen_image_slices" in batch[0] else None
        rope_image_info = [item["rope_image_info"] for item in batch] if "rope_image_info" in batch[0] else None

        ret = {
            "data_type": data_type,
            "dtype": dtype,
            "reward_tags": reward_tags,
            "text": text,
            "seeds": seeds,
            "n_samples": n_samples,
            "src_images": src_images,
            "use_face_rewards": use_face_rewards,
            "sem_points": sem_points,
            "subtask": subtask,
            "image": image,
            "lose_image": lose_image,
            "lose_exist": lose_exist,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            # "src_image_mask": src_image_mask,
            "image_mask": image_mask,
            # Conditional values
            "attention_mask": attention_mask,
            "iw_ih_scatter_index": iw_ih_scatter_index,  # (2)
            "iw_ih_scatter_src": iw_ih_scatter_src,      # (2)
            "timestep_scatter_index": timestep_scatter_index,   # (1)
            "rope_image_info": rope_image_info,  # (2)
            "gen_image_slices": gen_image_slices,
            # "src_face_embedding": src_face_embedding,
            # "src_image_slices": src_image_slices,
            # "und_image_slices": und_image_slices,
        }
        if "short_caption" in batch[0]:
            short_caption = [item["short_caption"] for item in batch]
            ret["short_caption"] = short_caption
        if "use_sft_loss" in batch[0]:
            ret["use_sft_loss"] = [item["use_sft_loss"] for item in batch]
        
        ret = {key: value for key, value in ret.items() if value is not None}

        return ret
