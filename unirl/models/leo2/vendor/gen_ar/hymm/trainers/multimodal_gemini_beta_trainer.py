import gc
import fnmatch
import json
from functools import partial
from typing import Dict, Union

import torch

from index_kits.sampler import DistributedSampler
from torch.utils.data import DataLoader
try:
    from torch.nn.attention.flex_attention import create_block_mask
except:
    pass

from hymm.trainers.transfusion_parallel import tp_sp_decorator

from ..utils.import_utils import require_version, is_index_kits_version
from ..utils.torch_utils import set_worker_seed_builder, to_device
from ..utils.helpers import multi_pattern_match
from ..data_kits.combined_iterator import CombinedBatchIterator
from ..data_kits.mmu_loader_v2 import MultiModalUnderstandingArrowStream
from ..data_kits.text_loader import TextArrowStream, MaxLengthBatchSampler
from ..data_kits.t2i_loader import (
    TextImageArrowStream, TextImageToImageArrowStream, TextImageInterleaveArrowStream
)
from ..models.autoregressive.flex_attn_layers import (
    create_interleave_mask_mod,
    create_batch_interleave_mask_mod,
    create_text_image_mask_mod,
    create_batch_text_image_mask_mod,
    create_causal_mask_mod,
    create_batch_causal_mask_mod,
)
from ..trainers.multimodal_gemini_alpha_trainer import GeminiTrainerAlphaMultiModal
from ..utils.torch_utils import DummyTensor

gc.set_threshold(7000, 100, 100)


def hy_text_length_getter(index_manager, ind):
    return index_manager.get_attribute(ind, "hy_ids_length")


@tp_sp_decorator
class MultiModalGeminiBetaTrainer(GeminiTrainerAlphaMultiModal):
    def task_init(self, args, all_dataset_keys=None):
        # 1. 把命令行里的 json 字符串转成真正的采样比例字典
        self.sampling_probs_dict = json.loads(args.sampling_probs)

        # 2. 保证“数据集 key → 整数 id”顺序与采样比例顺序完全一致，后面合并迭代器要用
        # The id should be consistent with the order of sampling_probs_dict.
        self.all_dataset_keys = list(self.sampling_probs_dict.keys())
        self.keys2id = {key: i for i, key in enumerate(self.all_dataset_keys)}
        print(f"Dataset keys-id mapping: {self.keys2id}")

        # 3. 定义“每种 dummy token 会出现在哪些任务里”，支持通配符（*）
        # Define what dummy token are incurred by each task.
        self.dummy_to_tasks = dict(
            t2i=["t2i*", "editing", "subject_driven", "interleave*", "face_id_clip", "editcot", "pair*"],  # iw, ih, timestep embedding, unet
            mmu=["mmu*", "face_id_clip"],  # iw, ih embedding, vision encoder, vision aligner
            face=["face_id_embedding"],  # qformer
        )

        # 4. 如果开启“联合图像特征”模式，把 pair/interleave 也划给 mmu 管
        self.use_joint_image_feature = args.get("use_joint_image_feature", False)
        if self.use_joint_image_feature:
            self.dummy_to_tasks['mmu'].extend(["pair*", "interleave*"])

        # 5. 计算每种任务实际要插入的 dummy token 数量
        # Set the number of dummy tokens
        self.dummy_dict = dict(
            t2i=1 + (2 if args.add_iw_ih_token else 0) + (1 if args.add_timestep_token else 0),  # 基础1 + 宽高2 + 步长1
            mmu=1 + (2 if args.add_iw_ih_token else 0),                                          # 基础1 + 宽高2
            face=args.get('face_aligner_num_queries', 16),                                       # 人脸 QFormer 默认16
        )

        # 6. 打印最终各任务 dummy 数量，方便肉眼核对
        for key, value in self.dummy_dict.items():
            print(f"Using {value} {key} dummy tokens.")

    def build_dataloader(self):
        args = self.args

        # ---------- 0. 父类里先把一些公共目录、worker 种子等预备好 ----------
        self.dataloader_preliminary_setup()

        # ---------- 1. 先准备两份“模板”配置，后面所有 DataLoader 复用 ----------
        dataloader_kwargs = dict(
            **args.dataloader_params,
            worker_init_fn=set_worker_seed_builder(self.dp_rank),
            shuffle=False,
            drop_last=True,
        )
        sampler_kwargs = dict(
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
        )

        # 如果用户想用 numpy 索引加速，强制 index-kits ≥ 0.5.11
        if args.get("use_numpy_indices", False):
            require_version("index-kits", "v0.5.11", "DistributedSampler with `use_numpy_indices` enabled")
            sampler_kwargs.update({"use_numpy_indices": True})

        # index-kits ≥ 0.5.14 支持在 sampler 里打印 dp/pp 信息，方便调试
        if is_index_kits_version(">=", "0.5.14"):
            if args.use_ptm:
                from megatron import mpu
                dp_rank = mpu.get_data_parallel_rank()
                pp_rank = mpu.get_pipeline_model_parallel_rank()
                info_repr = f"dp{dp_rank}, pp{pp_rank}"
            else:
                info_repr = ""
            sampler_kwargs.update({"verbose": 1, "info_repr": info_repr})

        # ---------- 2. 两个小工具函数 ----------
        def is_matched_task(task):
            '''看某个 dummy 任务（如 t2i*）是否出现在 all_dataset_keys 里'''
            matched = fnmatch.filter(self.all_dataset_keys, task)
            return len(matched) > 0

        def _filter_dummies(dummy_list):
            # Filter out dummies that are not in the dataset keys
            '''
                只保留“当前数据集里真的会出现”的 dummy 类型
            '''
            valid_dummies = []
            for dummy_candidate in dummy_list:
                if any(is_matched_task(task) for task in self.dummy_to_tasks[dummy_candidate]):
                    valid_dummies.append(dummy_candidate)
            return valid_dummies

        def find_matched_item(_dataset_tag, _task_info_dict):
            '''用通配符给 dataset_tag 找对应的任务配置（cur_task + 数据集类）'''
            item = None
            for key_pattern, item_ in _task_info_dict.items():
                if fnmatch.fnmatch(_dataset_tag, key_pattern):
                    item = item_
                    break
            return item

        # =====================================
        #     Text(+Image) to image data
        # =====================================
        self.task_dummy_dict['t2i'] = _filter_dummies(['mmu', 'face'])
        self.task_dummy_dict['ti2i'] = _filter_dummies(['mmu', 'face']) if not self.use_joint_image_feature else _filter_dummies(['face'])
        self.task_dummy_dict['face_id_clip'] = []
        self.task_dummy_dict['face_id_embedding'] = _filter_dummies(['mmu'])
        # Task info dictionary with wildcard support. Notice that only the tasks with the same
        # `cur_task` (for determining dummy type) and the same class can share the same entry.
        # 任务模板：通配符 -> 当前任务名 + 数据集类
        task_info_dict = {
            "t2i*": dict(cur_task="t2i", cls=TextImageArrowStream),
            "pair*": dict(cur_task="ti2i", cls=TextImageInterleaveArrowStream),
            "interleave*": dict(cur_task="ti2i", cls=TextImageInterleaveArrowStream),
            # bc
            "editing": dict(cur_task="ti2i", cls=TextImageToImageArrowStream),
            "editcot": dict(cur_task="ti2i", cls=TextImageToImageArrowStream),
            "subject_driven": dict(cur_task="ti2i", cls=TextImageToImageArrowStream),
            "face_id_clip": dict(cur_task="face_id_clip", cls=TextImageToImageArrowStream),
            "face_id_embedding": dict(cur_task="face_id_embedding", cls=TextImageToImageArrowStream),
        }

        # 遍历所有在采样比例里出现的数据集 key
        for dataset_tag in self.all_dataset_keys:
            item = find_matched_item(dataset_tag, task_info_dict)

            # 没在 task_info_dict 里匹配到模板，跳过
            if item is None:
                continue

            # 单任务调试模式，只建指定 key
            if self.cur_key is not None and dataset_tag != self.cur_key:
                continue

            # 已经建过了，跳过
            if dataset_tag in self.dataset_dict:
                continue

            # 读这个数据集专属 batch_size；没有就用全局 micro_batch_size
            task_batch_size = args.get(f'{dataset_tag}_batch_size', self.micro_batch_size)

            # 判断是否开“多分辨率”训练（multireso），决定 sampler 传不传 batch_size 进去
            if dataset_tag == "t2i":
                multireso = args["t2i_index_kwargs"]["multireso"]
            else:
                multireso = args[f'{dataset_tag}_index_kwargs'][f"{dataset_tag}_multireso"]
            
            # --------------- 真正实例化数据集 -----------------
            self.dataset_dict[dataset_tag] = item['cls'](
                args=args,
                dataset_tag=dataset_tag,
                tokenizer_name=args.tokenizer_name,
                task_kwargs=args.get(f'{dataset_tag}_task_kwargs'),
                index_kwargs=dict(
                    batch_size=task_batch_size if multireso else 1,  # Provide bsz to use multireso.
                    world_size=1,  # Dataset don't need to align with world_size. It will be handled by the sampler.
                    **args.get(f'{dataset_tag}_index_kwargs'),
                ),
                logger=self.logger,
                dummy_number=sum([self.dummy_dict[task] for task in self.task_dummy_dict[item['cur_task']]]),
                template=args.sequence_template,
            )
            # Build sampler and data loader
            self.sampler_dict[dataset_tag] = DistributedSampler(
                self.dataset_dict[dataset_tag],
                num_replicas=self.dataset_num_replicas[dataset_tag],
                rank=self.dataset_rank[dataset_tag],
                batch_size=task_batch_size if multireso else 1,  # Provide bsz to use multireso.
                **sampler_kwargs,
            )
            self.dataloader_dict[dataset_tag] = DataLoader(
                self.dataset_dict[dataset_tag],
                batch_size=task_batch_size,
                sampler=self.sampler_dict[dataset_tag],
                collate_fn=(self.dataset_dict[dataset_tag].collate_fn
                            if hasattr(self.dataset_dict[dataset_tag], "collate_fn")
                            else None),
                **dataloader_kwargs,
            )

        # =====================================
        #             Language data
        # =====================================
        self.task_dummy_dict['lm'] = _filter_dummies(['t2i', 'mmu', 'face'])
        lm_dummy_number = sum([self.dummy_dict[task] for task in self.task_dummy_dict['lm']])
        lm_task_info_dict = {
            "lm*": dict(cur_task="lm", cls=TextArrowStream),
        }
        for dataset_tag in self.all_dataset_keys:
            item = find_matched_item(dataset_tag, lm_task_info_dict)
            if item is None:
                continue
            if self.cur_key is not None and dataset_tag != self.cur_key:
                continue
            if dataset_tag in self.dataset_dict:
                continue

            lm_batch_size = args.get(f'{dataset_tag}_batch_size', self.micro_batch_size)
            lm_dataset_kwargs = dict(
                args=args,
                max_token_length=None,
                tokenizer_name=args.tokenizer_name,
                task_kwargs=args.get(f'{dataset_tag}_task_kwargs'),
                index_kwargs=args.get(f'{dataset_tag}_index_kwargs'),
                logger=self.logger,
                single_text_max_length=args.get(f'{dataset_tag}_single_text_max_length', 100000),   # bc
                pre_extract_tokens=args.get(f'{dataset_tag}_pre_extract_tokens', False),    # bc
                dummy_number=lm_dummy_number,
                token_length=args.get(f"{dataset_tag}_token_length"),   # bc
            )

            # ------------- LM ---------------
            self.dataset_dict[dataset_tag] = item['cls'](
                index_file=args.get(f'{dataset_tag}_index_file'),
                dataset_type=dataset_tag,
                template=args.sequence_template,
                **lm_dataset_kwargs,
            )
            self.sampler_dict[dataset_tag] = DistributedSampler(
                self.dataset_dict[dataset_tag],
                num_replicas=self.dataset_num_replicas[dataset_tag],
                rank=self.dataset_rank[dataset_tag],
                **sampler_kwargs,
            )
            if self.dataset_dict[dataset_tag].sequence_pack:
                self.dataloader_dict[dataset_tag] = DataLoader(
                    self.dataset_dict[dataset_tag],
                    batch_size=lm_batch_size,
                    sampler=self.sampler_dict[dataset_tag],
                    collate_fn=(self.dataset_dict[dataset_tag].collate_fn
                                if hasattr(self.dataset_dict[dataset_tag], "collate_fn")
                                else None),
                    **dataloader_kwargs,
                )
            elif args.sequence_template == "pretrain":
                self.batch_sampler_dict[dataset_tag] = MaxLengthBatchSampler(
                    self.dataset_dict[dataset_tag].index_manager,
                    self.sampler_dict[dataset_tag],
                    length_getter=hy_text_length_getter,
                    batch_size=lm_batch_size,
                    max_length=self.dataset_dict[dataset_tag].token_length,
                )
                self.dataloader_dict[dataset_tag] = DataLoader(
                    self.dataset_dict[dataset_tag], batch_sampler=self.batch_sampler_dict[dataset_tag],
                    worker_init_fn=set_worker_seed_builder(self.dp_rank),
                    **args.dataloader_params,
                )
            else:
                self.dataloader_dict[dataset_tag] = DataLoader(
                    self.dataset_dict[dataset_tag],
                    batch_size=lm_batch_size,
                    sampler=self.sampler_dict[dataset_tag],
                    **dataloader_kwargs,
                )
        # ================================================
        #          Multimodal understanding data
        # ================================================
        mmu_task_info_dict = {
            "mmu*": dict(cur_task="mmu", cls=MultiModalUnderstandingArrowStream),
        }
        # Set the number of dummy tokens for mmu sequences
        self.task_dummy_dict['mmu'] = _filter_dummies(['t2i', 'face'])
        for dataset_tag in self.all_dataset_keys:
            item = find_matched_item(dataset_tag, mmu_task_info_dict)
            if item is None:
                continue
            if self.cur_key is not None and dataset_tag != self.cur_key:
                continue
            if dataset_tag in self.dataset_dict:
                continue

            mmu_batch_size = args.get(f'{dataset_tag}_batch_size', self.micro_batch_size)
            self.dataset_dict[dataset_tag] = item['cls'](
                args=args,
                dataset_tag=dataset_tag,
                tokenizer_name=args.tokenizer_name,
                task_kwargs=args.get(f'{dataset_tag}_task_kwargs'),
                index_kwargs=args.get(f'{dataset_tag}_index_kwargs'),
                template=args.sequence_template,
                dummy_number=sum([self.dummy_dict[task] for task in self.task_dummy_dict[item['cur_task']]]),
                logger=self.logger,
            )
            self.sampler_dict[dataset_tag] = DistributedSampler(
                self.dataset_dict[dataset_tag],
                num_replicas=self.dataset_num_replicas[dataset_tag],
                rank=self.dataset_rank[dataset_tag],
                **sampler_kwargs,
            )
            self.dataloader_dict[dataset_tag] = DataLoader(
                self.dataset_dict[dataset_tag],
                batch_size=mmu_batch_size,
                sampler=self.sampler_dict[dataset_tag],
                collate_fn=(self.dataset_dict[dataset_tag].collate_fn
                            if hasattr(self.dataset_dict[dataset_tag], "collate_fn")
                            else None),
                **dataloader_kwargs,
            )

        # ================================================
        #                  Sanity Check
        # ================================================
        # Make sure all the dataset keys are initialized
        if self.cur_key is None:
            if len(self.all_dataset_keys) != len(self.dataset_dict):
                missing_keys = []
                unexpected_keys = []
                for key in self.all_dataset_keys:
                    if key not in self.dataset_dict:
                        missing_keys.append(key)
                for key in self.dataset_dict.keys():
                    if key not in self.all_dataset_keys:
                        unexpected_keys.append(key)
                raise ValueError(f"Dataset keys mismatch.\n"
                                 f"not implemented dataset keys: {missing_keys}\n"
                                 f"unexpected dataset keys: {unexpected_keys}")
        else:
            if len(self.dataset_dict) != 1:
                raise ValueError(
                    f"Expected only one dataset key, but got {len(self.dataset_dict)}: {self.dataset_dict.keys()}. "
                    f"Current key is {self.cur_key}, but dataset_dict has keys {self.dataset_dict.keys()}."
                )
            else:
                assert list(self.dataset_dict.keys())[0] == self.cur_key, \
                    f"Current key {self.cur_key} does not match dataset_dict keys {self.dataset_dict.keys()}."

    def build_data_iterator(self, **kwargs):
        self.dataloader = CombinedBatchIterator(
            ss=self.ss,
            fast_shuffle=self.args.fast_shuffle,
            rank=self.dp_rank,
            world_size=self.dp_size,
            datasets=self.dataset_dict,
            samplers=self.sampler_dict,
            dataloaders=self.dataloader_dict,
            sampling_probs=self.sampling_probs_dict,
            initial_seed=self.args.global_seed,
            sampling_mode=self.args.get('combined_iterator_sampling_mode', 'random'),
            cache_shuffle=self.args.get('cache_shuffle'),
            fixed_key=self.cur_key,
            fixed_key_group=self.cur_key_group,
            **kwargs,
            force_sync_shuffle=False,
            keys2id=self.keys2id,
        )

    @staticmethod
    def print_batch(batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}, {value.dtype}")
            elif isinstance(value, list):
                print(f"{key}: [", end="")
                for v in value:
                    if isinstance(v, torch.Tensor):
                        print(f"({v.shape}, {v.dtype}), ", end="")
                    elif isinstance(v, list):
                        print("[", end="")
                        for sv in v:
                            if isinstance(sv, torch.Tensor):
                                print(f"({sv.shape}, {sv.dtype}), ", end="")
                            else:
                                print(f"({type(sv)}), ", end="")
                        print("], ", end="")
                print("]")
            else:
                print(f"{key}: {type(value)}")

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        # batch["dtype"] are actually dataset_tag.
        if multi_pattern_match(batch["dtype"][0], ["t2i*"]):
            inputs = self.prepare_model_t2i_inputs(batch, device, **kwargs)

        elif multi_pattern_match(batch["dtype"][0], ["pair*", "edit*", "interleave*", "subject_driven"]):
            inputs = self.prepare_model_ti2i_inputs(batch, device, **kwargs)

        elif multi_pattern_match(batch["dtype"][0], ["lm*"]):
            inputs = self.prepare_model_lm_inputs(batch, device, **kwargs)

        elif multi_pattern_match(batch["dtype"][0], ["mmu*"]):
            inputs = self.prepare_model_mmu_inputs(batch, device, **kwargs)

        elif multi_pattern_match(batch["dtype"][0], ["face_id*"]):
            inputs = self.prepare_model_face_id_clip_inputs(batch, device, **kwargs)

        else:
            raise ValueError(f"Unknown batch dtype, expected {self.all_dataset_keys}, got {batch['dtype']}")
        return inputs

    def prepare_model_t2i_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)
        batch_size, n_tokens = tokens.shape

        # Add dummy tokens
        extra = dict(
            n_samples=to_device(batch["n_samples"], device),  # 样本数量
            text_mask=text_mask,        # [b, seqlen]
            image_mask=image_mask,      # [b, seqlen]
        )

        # 如果batch中包含rope_image_info，添加到extra中
        if "rope_image_info" in batch:
            extra.update(dict(
                rope_image_info=batch["rope_image_info"],
            ))
        
        # 处理偏移量信息
        if "offsets" in batch:
            extra.update(dict(
                # 使用clamp限制偏移量范围，避免边界情况。因为n_tokens = batch["tokens"].shape[1] - 1
                sample_offsets=[
                    # clamp by n_tokens to avoid the boundary situation where offset[-1] == batch["tokens"].shape[1]
                    # due to n_tokens = batch["tokens"].shape[1] - 1
                    torch.clamp(offset, 0, n_tokens)
                    for offset in batch["offsets"]
                ],
            ))
        if "iw_ih_scatter_index" in batch:
            extra.update(dict(
                iw_ih_scatter_index=to_device(batch["iw_ih_scatter_index"], device),        # [b, 2]
                iw_ih_scatter_src=to_device(batch["iw_ih_scatter_src"], device),            # [b, 2]
            ))
        if "timestep_scatter_index" in batch:
            extra.update(dict(
                timestep_scatter_index=to_device(batch["timestep_scatter_index"], device),  # [b, 1]
            ))

        # 为各种任务添加虚拟token
        for task in self.task_dummy_dict['t2i']:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )

        # ===================================== 准备注意力掩码 =====================================
        # 根据任务类型和配置选择注意力掩码类型
        # Attention mask
        if batch["dtype"][0] == "t2i":
            attn_type = self.args.get(f't2i_task_kwargs', {}).get('attn_type', 'auto')
        else:
            attn_type = self.args.get(f'{batch["dtype"][0]}_task_kwargs', {}).get(f'{batch["dtype"][0]}_attn_type', 'auto')
        
        # 使用预定义的注意力掩码
        if attn_type == 'auto':
            attention_mask = batch["attention_mask"].to(device)

        # 使用灵活的注意力掩码（需要序列长度能被128整除）
        elif attn_type == 'flex':
            bsz, seq_len = tokens.shape
            assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."

            # 单批次情况：为单个样本创建文本-图像掩码
            if bsz == 1:
                image_slices = batch["gen_image_slices"][0]
                mask_mod = create_text_image_mask_mod(
                    image_slices, seq_len, device,
                    offsets=extra.get("sample_offsets", [None])[0],
                )
                attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

            # 多批次情况：为整个批次创建文本-图像掩码
            else:
                mask_mod = create_batch_text_image_mask_mod(
                    batch["gen_image_slices"], seq_len, device,
                    batch_offsets=extra.get("sample_offsets"),
                )
                attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

            # 设置注意力掩码的数据类型和设备
            attention_mask.dtype = torch.bool
            attention_mask.device = tokens.device
        else:
            raise NotImplementedError(f"Attention type {attn_type} is not supported.")

        # ===================================== prepare diffusion =====================================
        if kwargs.get('skip_vae_encode'):
            t, model_t, x_0, x_t, u_t = None, None, None, None, None
        else:
            # 使用VAE对图像进行编码，获取扩散过程所需的各种变量
            out = self.vae_encode(batch["image"], sample_type="sample")
            t, model_t, x_0, x_t, u_t = out.t, out.model_t, out.x_0, out.x_t, out.u_t

        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                         # [b, seqlen] 输入token序列
            x_t=x_t,                            # [b, c, h, w] 噪声图像（扩散过程的时间步t）
            t=model_t,                          # [b] 时间步
            diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
            target=target_tokens,               # [b, seqlen] 目标token序列（用于计算loss）
            attention_mask=attention_mask,      # [b, seqlen, seqlen] 
            image_loss_weight=self.args.image_loss_weight,      # 图像损失权重
            data_type=batch['data_type'][0],                    # 数据类型（用于loss计算）
            **extra,
        )

        # 保存第一个训练样本的输入参数和相关信息（用于调试或后续分析）
        # Save model_input_kwargs and a few more info for reconstructing the attention mask
        self.save_first_training_samples(model_intput_kwargs, extra=dict(
            gen_image_slices=batch.get("gen_image_slices", None),  # 生成的图像切片信息
        ))
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_ti2i_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

        # 处理源图像掩码（对于interleave数据可能为None）
        # For interleave data, src_image_mask and src_images can be None
        if "src_image_mask" in batch:
            src_image_mask = batch["src_image_mask"][:, :-1].contiguous().to(device)
        else:
            src_image_mask = None
        
        # 处理图像掩码（对于interleave数据可能为None）
        if "und_image_mask" in batch:
            und_image_mask = batch["und_image_mask"][:, :-1].contiguous().to(device)
        else:
            und_image_mask = None
        
        # 获取批次大小和token数量
        batch_size, n_tokens = tokens.shape

        # 添加额外的输入参数
        # Add dummy tokens
        extra = dict(
            n_samples=to_device(batch["n_samples"], device),  # 样本数量
            text_mask=text_mask,            # [b, seqlen] 文本掩码
            image_mask=image_mask,          # [b, seqlen] 生成图像掩码
            **(dict(src_image_mask=src_image_mask) if src_image_mask is not None else {}),  # [b, seqlen] 源图像掩码
            **(dict(und_image_masks=und_image_mask) if und_image_mask is not None else {}),  # [b, seqlen] 图像掩码
        )

        # 如果batch中包含rope_image_info，添加到extra中
        if "rope_image_info" in batch:
            extra.update(dict(
                rope_image_info=batch["rope_image_info"],
            ))

        # 处理偏移量信息
        if "offsets" in batch:
            extra.update(dict(
                sample_offsets=[
                    # 使用clamp限制偏移量范围，避免边界情况
                    # clamp by n_tokens to avoid the boundary situation where offset[-1] == batch["tokens"].shape[1]
                    # due to n_tokens = batch["tokens"].shape[1] - 1
                    torch.clamp(offset, 0, n_tokens)
                    for offset in batch["offsets"]
                ],
            ))

        # 处理图像宽高索引
        if "iw_ih_scatter_index" in batch:
            extra.update(dict(
                iw_ih_scatter_index=to_device(batch["iw_ih_scatter_index"], device),        # [b, 2]
                iw_ih_scatter_src=to_device(batch["iw_ih_scatter_src"], device),            # [b, 2]
            ))

        # 处理时间步索引
        if "timestep_scatter_index" in batch:
            extra.update(dict(
                timestep_scatter_index=to_device(batch["timestep_scatter_index"], device),  # [b, 1]
            ))

        # 为ti2i任务添加虚拟token
        for task in self.task_dummy_dict['ti2i']:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )

        # Attention mask
        # 缓存清理标志
        if not hasattr(self, '_flex_cache_cleared'):
            self._flex_cache_cleared = False
        
        # 根据数据集类型选择注意力类型
        dataset_tag = batch["dtype"][0]
        attn_type = self.args.get(f'{dataset_tag}_task_kwargs', {}).get(f'{dataset_tag}_attn_type', 'auto')

        # 使用预定义的注意力掩码
        if attn_type == 'auto':
            attention_mask = batch["attention_mask"].to(device)

        # 使用灵活的注意力掩码
        elif attn_type == 'flex':
            bsz, seq_len = tokens.shape
            assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."

            # 计算图像前缀和后缀token数量
            num_image_prefix = (
                    1 +                                              # 基本图像前缀
                    (2 if self.args.add_image_shape_token else 0) +  # 图像形状token
                    (1 if self.args.add_timestep_token else 0)       # 时间步token
            )
            num_image_suffix = 1  # 图像后缀token数量

            # 处理interleave数据类型的注意力掩码
            if batch["dtype"][0] == "interleave":

                # 单批次数据
                if bsz == 1:
                    if self.args.use_joint_image_feature:
                        cond_slices = batch["joint_image_slices"][0]  # 使用联合图像特征
                    else:
                        cond_slices = batch["src_image_slices"][0] + batch["und_image_slices"][0]  # 源图像+图像

                    gen_image_slices = batch["gen_image_slices"][0]  # 生成图像切片
                    hole_slices = []

                    # 创建空洞切片：除了最后一个生成图像，其他图像不允许被后续token关注
                    for sli in gen_image_slices:
                        # Gen images except the last one are not allowed to be attended by following tokens.
                        if tokens[0, sli.stop + 1] != self.tkwrapper.eos_token:
                            hole_slices.append(slice(sli.start - num_image_prefix, sli.stop + num_image_suffix))
                    
                    mask_mod = create_interleave_mask_mod(
                        gen_image_slices, cond_slices, hole_slices, seq_len, device,
                        offsets=extra.get("sample_offsets", [None])[0],
                    )
                    attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

                # 多批次数据
                else:
                    if self.args.use_joint_image_feature:
                        cond_slices = batch["joint_image_slices"]  # 使用联合图像特征
                    else:
                        cond_slices = [
                            batch["src_image_slices"][i] + batch["und_image_slices"][i]
                            for i in range(bsz)
                        ]
                    
                    gen_image_slices = batch["gen_image_slices"]  # 生成图像切片
                    hole_slices = []

                    # 为每个批次创建空洞切片
                    for gen_image_slices_i in gen_image_slices:
                        hole_slices_i = []
                        for sli in gen_image_slices_i:
                            # Gen images except the last one are not allowed to be attended by following tokens.
                            if tokens[0, sli.stop + 1] != self.tkwrapper.eos_token:
                                hole_slices_i.append(slice(sli.start - num_image_prefix, sli.stop + num_image_suffix))
                        hole_slices.append(hole_slices_i)
                    
                    mask_mod = create_batch_interleave_mask_mod(
                        gen_image_slices, cond_slices, hole_slices, seq_len, device,
                        batch_offsets=extra.get("sample_offsets"),
                    )

                    attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

            # 处理非interleave数据类型的注意力掩码
            elif bsz == 1:
                if self.args.use_joint_image_feature:
                    image_slices = batch["gen_image_slices"][0] + batch["joint_image_slices"][0]
                else:
                    image_slices = batch["gen_image_slices"][0] + batch["src_image_slices"][0] + batch["und_image_slices"][0]
                
                mask_mod = create_text_image_mask_mod(image_slices, seq_len, device,
                                                      offsets=extra.get("sample_offsets", [None])[0])
                attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
            else:
                if self.args.use_joint_image_feature:
                    batch_image_slices = [
                        batch["gen_image_slices"][i] + batch["joint_image_slices"][i]
                        for i in range(bsz)
                    ]
                else:
                    batch_image_slices = [
                        batch["gen_image_slices"][i] + batch["src_image_slices"][i] + batch["und_image_slices"][i]
                        for i in range(bsz)
                    ]
                
                mask_mod = create_batch_text_image_mask_mod(batch_image_slices, seq_len, device,
                                                            batch_offsets=extra.get("sample_offsets"))
                attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

            # 设置注意力掩码属性
            attention_mask.dtype = torch.bool
            attention_mask.device = tokens.device
        else:
            raise NotImplementedError(f"Attention type {attn_type} is not supported.")

        # 清理GPU缓存（仅第一次执行时）
        if not self._flex_cache_cleared:
            torch.cuda.empty_cache()
            self._flex_cache_cleared = True

        # ===================================== prepare diffusion =====================================
        # VAE缓存清理标志
        if not hasattr(self, '_vae_cache_cleared'):
            self._vae_cache_cleared = False

        # 跳过VAE编码
        if kwargs.get('skip_vae_encode'):
            t, model_t, x_0, x_t, u_t = None, None, None, None, None
            input_src_t, input_src_x = None, None

        else:
            # 对目标图像进行VAE编码
            out = self.vae_encode(batch["image"], sample_type="sample")
            t, model_t, x_0, x_t, u_t = out.t, out.model_t, out.x_0, out.x_t, out.u_t

            # 对源图像进行VAE编码
            if "src_images" in batch and batch["src_images"] is not None:
                sout = self.vae_encode(batch["src_images"], sample_type="sample_start")
                input_src_t, input_src_x = sout.model_t, sout.x_t
            else:
                input_src_t, input_src_x = None, None

        # 清理VAE编码后的GPU缓存（仅第一次执行时）
        if not self._vae_cache_cleared:
            torch.cuda.empty_cache()
            self._vae_cache_cleared = True

        # 处理联合图像模式
        # joint image mode
        if "und_images" in batch:
            und_images = to_device(batch["und_images"], device)
            assert "und_images" not in extra, "und_images should not be added for dummy token"
            extra.update(dict(
                und_images=und_images,
            ))

        # 处理视觉编码器参数
        if "vision_encoder_kwargs" in batch:
            try:
                # 尝试直接转换整个字典
                vision_encoder_kwargs = {k: to_device(v, device) for k, v in batch["vision_encoder_kwargs"].items()}
            except Exception as e:
                # 如果失败，对每个值进行逐元素转换
                vision_encoder_kwargs = {k: [to_device(v_, device) for v_ in v] for k, v in batch["vision_encoder_kwargs"].items()}
            
            extra.update(dict(
                vision_encoder_kwargs=vision_encoder_kwargs,  # 视觉编码器参数
            ))

        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                         # [b, seqlen] 输入token序列
            x_t=x_t,                            # [b, c, h, w] 噪声图像（目标图像）
            t=model_t,                          # [b] 时间步
            diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
            src_x=input_src_x,                  # [b, c, h, w] 源图像
            src_t=input_src_t,                  # [b] 源图像时间步
            target=target_tokens,               # [b, seqlen] 目标token序列
            attention_mask=attention_mask,      # [b, seqlen, seqlen] 注意力掩码
            image_loss_weight=self.args.image_loss_weight,
            data_type=batch['data_type'][0],                    # 数据类型（用于loss计算）
            **extra,
        )

        # 保存第一个训练样本的输入参数和相关信息
        self.save_first_training_samples(model_intput_kwargs, extra=dict(
            gen_image_slices=batch.get("gen_image_slices", None),  # 生成图像切片
            src_image_slices=batch.get("src_image_slices", None),  # 源图像切片
            und_image_slices=batch.get("und_image_slices", None),  # 图像切片
            joint_image_slices=batch.get("joint_image_slices", None),  # 联合图像切片
        ))
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_face_id_clip_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)
        dataset_tag = batch["dtype"][0]
        if hasattr(self.args, f"{dataset_tag}_local_loss_weight"):
            token_bbox_mask = batch["token_bbox_mask"].to(device)
        else:
            token_bbox_mask = None

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,            # [b, seqlen]
            image_mask=image_mask,          # [b, seqlen]
        )
        if "rope_image_info" in batch:
            extra.update(dict(
                rope_image_info=batch["rope_image_info"],
            ))
        if "iw_ih_scatter_index" in batch:
            extra.update(dict(
                iw_ih_scatter_index=to_device(batch["iw_ih_scatter_index"], device),        # [b, 2]
                iw_ih_scatter_src=to_device(batch["iw_ih_scatter_src"], device),            # [b, 2]
            ))
        if "timestep_scatter_index" in batch:
            extra.update(dict(
                timestep_scatter_index=to_device(batch["timestep_scatter_index"], device),  # [b, 1]
            ))
        for task in self.task_dummy_dict[dataset_tag]:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )
        batch_size, n_tokens = tokens.shape

        # Attention mask
        attn_type = self.args.get(f'{dataset_tag}_task_kwargs', {}).get(f'{dataset_tag}_attn_type', 'auto')
        if attn_type == 'auto':
            attention_mask = batch["attention_mask"].to(device)
        elif attn_type == 'flex':
            bsz, seq_len = tokens.shape
            assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."
            if bsz == 1:
                image_slices = batch["gen_image_slices"][0] + batch["src_image_slices"][0] + batch["und_image_slices"][0] + batch["joint_image_slices"][0]
                if "face_image_slices" in batch:
                    image_slices += batch["face_image_slices"][0]
                mask_mod = create_text_image_mask_mod(image_slices, seq_len, device)
                attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
            else:
                # Add face_embedding_slices here
                if "face_image_slices" in batch:
                    batch_image_slices = [
                        batch["gen_image_slices"][i] + batch["src_image_slices"][i] + batch["und_image_slices"][i] + batch["joint_image_slices"][i] + batch["face_image_slices"][i]
                        for i in range(bsz)
                    ]
                else:
                    batch_image_slices = [
                        batch["gen_image_slices"][i] + batch["src_image_slices"][i] + batch["und_image_slices"][i] + batch["joint_image_slices"][i]
                        for i in range(bsz)
                    ]
                mask_mod = create_batch_text_image_mask_mod(batch_image_slices, seq_len, device)
                attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
            attention_mask.dtype = torch.bool
            attention_mask.device = tokens.device
        else:
            raise NotImplementedError(f"Attention type {attn_type} is not supported.")

        # ===================================== prepare diffusion =====================================
        if kwargs.get('skip_vae_encode'):
            t, model_t, x_0, x_t, u_t = None, None, None, None, None
        else:
            out = self.vae_encode(batch["image"], sample_type="sample")
            t, model_t, x_0, x_t, u_t = out.t, out.model_t, out.x_0, out.x_t, out.u_t

            # Add face embedding to extra dict here; set und_images to avoid siglip condition
            condition_type = self.args.get(f"{dataset_tag}_condition_type", "clip")
            if condition_type == "face_embedding":
                extra.update(dict(
                    src_face_embedding=to_device(batch["src_face_embedding"], device),  # [b, 512]
                    und_image_masks=batch["face_image_mask"][:, :-1].contiguous().to(device),  # [b, seqlen]
                ))
                # und_images = None
            elif condition_type == "clip":
                extra.update(dict(
                    und_images=to_device(batch["src_images"], device),
                    und_image_masks=batch["und_image_mask"][:, :-1].contiguous().to(device),
                ))
            elif condition_type == "joint_image":
                # face task with joint_image condition has the same sequence composition as editing task.
                sout = self.vae_encode(batch["src_images"], sample_type="sample_start")
                input_src_t, input_src_x = sout.model_t, sout.x_t
                extra.update(dict(
                    src_x=input_src_x,
                    src_t=input_src_t,
                    src_image_mask=batch["src_image_mask"][:, :-1].contiguous().to(device),
                    und_images=to_device(batch["und_images"], device),
                    und_image_masks=batch["und_image_mask"][:, :-1].contiguous().to(device),
                ))
                if "vision_encoder_kwargs" in batch:
                    try:
                        vision_encoder_kwargs = {k: to_device(v, device) for k, v in batch["vision_encoder_kwargs"].items()}
                    except Exception as e:
                        vision_encoder_kwargs = {k: [to_device(v_, device) for v_ in v] for k, v in batch["vision_encoder_kwargs"].items()}
                    extra.update(dict(
                        vision_encoder_kwargs=vision_encoder_kwargs,
                    ))
            elif condition_type in ["face_embedding_src_image", "src_image_sep_face_embedding"]:
                sout = self.vae_encode(batch["src_images"], sample_type="sample_start")
                input_src_t, input_src_x = sout.model_t, sout.x_t
                extra.update(dict(
                    src_face_embedding=to_device(batch["src_face_embedding"], device),  # [b, 512]
                    und_image_masks=batch["face_image_mask"][:, :-1].contiguous().to(device),  # [b, seqlen]
                ))
                extra.update(dict(
                    src_x=input_src_x,                  # [b, c, h, w]
                    src_t=input_src_t,                  # [b]
                    src_image_mask=batch["src_image_mask"][:, :-1].contiguous().to(device),
                ))
            else:
                raise NotImplementedError(f"Condition type {condition_type} is not supported.")
        
        if hasattr(self.args, f"{dataset_tag}_local_loss_weight"):
            diffusion_loss_fn = partial(self.prepare_training_losses_fn_with_token_mask,
                                        t=t, x0=x_0, xt=x_t, ut=u_t, token_bbox_mask=token_bbox_mask,
                                        dataset_tag=dataset_tag)
        else:
            diffusion_loss_fn = partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t)
        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                         # [b, seqlen]
            x_t=x_t,                            # [b, c, h, w]
            t=model_t,                          # [b]
            diffusion_loss_fn=diffusion_loss_fn,
            target=target_tokens,               # [b, seqlen]
            attention_mask=attention_mask,      # [b, seqlen, seqlen]
            image_loss_weight=self.args.image_loss_weight,
            data_type=batch['data_type'][0],                    # For loss
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs, key_not_save=["diffusion_loss_fn"])
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_training_losses_fn_with_token_mask(self, t, x0, xt, ut, token_bbox_mask, model_output, dataset_tag):
        # model_t: t * self.training_timesteps
        # x_0:     random noise
        # x_t:     ICPlan.compute_mu_t = (1-t) x1 + t x0
        # u_t:     ICPlan.plan = (-1) x1 + (1) x0
        local_loss_weight = self.args.get(f"{dataset_tag}_local_loss_weight", 10)
        global_loss_weight = 1
        global_loss = self.denoiser.training_losses_fn(t, x0, xt, ut, model_output)["loss"]
        local_loss = self.denoiser.training_losses_fn(
            t, x0*token_bbox_mask, xt*token_bbox_mask, ut*token_bbox_mask, model_output*token_bbox_mask)["loss"]
        total_loss = global_loss_weight*global_loss + local_loss_weight*local_loss
        self.logger.info(f"[prepare_training_losses_fn_with_token_mask] ; "
                         f"face loss weight: {local_loss_weight}; global_loss: {global_loss}, "
                         f"local_loss: {local_loss}, total_loss: {total_loss}")
        return {"loss": total_loss}

    def prepare_model_lm_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        batch_size, n_tokens = tokens.shape

        # Add dummy tokens
        extra = dict(
            n_samples=to_device(batch["n_samples"], device),
            text_mask=text_mask,
        )
        if "offsets" in batch:
            extra.update(dict(
                sample_offsets=[
                    # clamp by n_tokens to avoid the boundary situation where offset[-1] == batch["tokens"].shape[1]
                    # due to n_tokens = batch["tokens"].shape[1] - 1
                    torch.clamp(offset, 0, n_tokens)
                    for offset in batch["offsets"]
                ],
            ))
        for task in self.task_dummy_dict['lm']:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )

        # Attention mask
        attn_type = self.args.get('lm_attn_type', 'auto')
        if attn_type == 'auto':
            causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
            attention_mask = causal_mask.view(1, 1, n_tokens, n_tokens).repeat(batch_size, 1, 1, 1)
        elif attn_type == 'flex':
            bsz, seq_len = tokens.shape
            assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."
            if bsz == 1:
                mask_mod = create_causal_mask_mod(seq_len, device, offsets=extra.get("sample_offsets", [None])[0])
                attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
            else:
                mask_mod = create_batch_causal_mask_mod(seq_len, device, batch_offsets=extra.get("sample_offsets"))
                attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
            attention_mask.dtype = torch.bool
            attention_mask.device = tokens.device
        elif attn_type == 'ptm_fa3':
            # Create an indices list (just named as attention_mask, no real attention mask) with shape [1, seq_len, 2]
            # This indices list is used to create cu_seqlens for FA inputs.
            attention_mask = torch.stack([
                torch.arange(n_tokens, dtype=torch.int32),  # Placeholder, no meaning
                torch.tensor([n_tokens - 1] * n_tokens, dtype=torch.int32)
            ], dim=1).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        elif attn_type == 'ptm_sdpa':
            attention_mask = DummyTensor(
                shape=[batch_size, n_tokens, n_tokens], dtype=torch.bool, device=tokens.device
            )
        else:
            # Default to causal
            attention_mask = None

        model_intput_kwargs = dict(
            idx=tokens,  # [b, 512]
            target=target_tokens,  # [b, 512]
            attention_mask=attention_mask,  # [b, 512, 512]
            image_loss_weight=0,    # Set to zero to avoid dummy image tokens to affect the text loss
            data_type="lm",         # For loss
            **extra,    # x_t, t, image_mask, und_images, und_image_masks, diffusion_loss_fn, iw, ih, timestep
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_mmu_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # und_image_mask is used to fill image_embeds, therefore should be shifted same as tokens
        und_image_mask = batch["und_image_mask"][:, :-1].contiguous().to(device)
        if "src_image_mask" in batch:
            src_image_mask = batch["src_image_mask"][:, :-1].contiguous().to(device)
        else:
            src_image_mask = None
        batch_size, n_tokens = tokens.shape

        # Add dummy tokens
        extra = dict(
            n_samples=to_device(batch["n_samples"], device),
            text_mask=text_mask,        # [b, seqlen]
            und_image_masks=und_image_mask,  # [b, seqlen]
            **(dict(src_image_mask=src_image_mask) if src_image_mask is not None else {}),  # [b, seqlen]
        )
        if "rope_image_info" in batch:
            extra.update(dict(
                rope_image_info=batch["rope_image_info"],
            ))
        if "offsets" in batch:
            extra.update(dict(
                sample_offsets=[
                    # clamp by n_tokens to avoid the boundary situation where offset[-1] == batch["tokens"].shape[1]
                    # due to n_tokens = batch["tokens"].shape[1] - 1
                    torch.clamp(offset, 0, n_tokens)
                    for offset in batch["offsets"]
                ],
            ))
        if "iw_ih_scatter_index" in batch:
            extra.update(dict(
                iw_ih_scatter_index=to_device(batch["iw_ih_scatter_index"], device),        # bsz x 2n
                iw_ih_scatter_src=to_device(batch["iw_ih_scatter_src"], device),            # bsz x 2n
            ))
        if "timestep_scatter_index" in batch:
            extra.update(dict(
                timestep_scatter_index=to_device(batch["timestep_scatter_index"], device),
            ))
        for task in self.task_dummy_dict['mmu']:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )

        # build attention mask. Intra-image is full attention, inter-image and image-text is causal attention.
        dataset_tag = batch["dtype"][0]
        attn_type = self.args.get(f'{dataset_tag}_task_kwargs', {}).get(f'{dataset_tag}_attn_type', 'auto')
        if attn_type == "auto":
            attention_mask = batch["attention_mask"].to(device)
        elif attn_type == "flex":
            bsz, seq_len = tokens.shape
            assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."
            if bsz == 1:
                if self.use_joint_image_feature:
                    image_slices = batch["joint_image_slices"][0]
                else:
                    image_slices = batch["src_image_slices"][0] + batch["und_image_slices"][0]
                mask_mod = create_text_image_mask_mod(image_slices, seq_len, device,
                                                      offsets=extra.get("sample_offsets", [None])[0])
                attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
            else:
                if self.args.use_joint_image_feature:
                    cond_slices = batch["joint_image_slices"]
                else:
                    cond_slices = [
                        batch["src_image_slices"][i] + batch["und_image_slices"][i]
                        for i in range(bsz)
                    ]
                mask_mod = create_batch_text_image_mask_mod(cond_slices, seq_len, device,
                                                            batch_offsets=extra.get("sample_offsets"))
                attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
            attention_mask.dtype = torch.bool
            attention_mask.device = tokens.device
        else:
            raise NotImplementedError(f"Attention type {attn_type} is not supported.")

        und_images = to_device(batch["und_images"], device)
        if "vision_encoder_kwargs" in batch:
            vision_encoder_kwargs = {k: to_device(v, device) for k, v in batch["vision_encoder_kwargs"].items()}
        else:
            vision_encoder_kwargs = None

        if "src_images" in batch and not kwargs.get('skip_vae_encode', False):
            sout = self.vae_encode(batch["src_images"], sample_type="sample_start")
            input_src_t, input_src_x = sout.model_t, sout.x_t
        else:
            input_src_t, input_src_x = None, None
        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 512]
            target=target_tokens,  # [b, 512]
            attention_mask=attention_mask,  # [b, 512, 512]
            image_loss_weight=0,    # Set to zero to avoid dummy image tokens to affect the text loss
            data_type="mmu",        # For loss
            und_images=und_images,
            vision_encoder_kwargs=vision_encoder_kwargs,
            src_x=input_src_x,                  # [b, c, h, w]
            src_t=input_src_t,                  # [b]
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs, extra=dict(
            und_image_slices=batch.get("und_image_slices", None),
            src_image_slices=batch.get("src_image_slices", None),
            joint_image_slices=batch.get("joint_image_slices", None),
        ))
        return model_intput_kwargs, batch_size, n_tokens
