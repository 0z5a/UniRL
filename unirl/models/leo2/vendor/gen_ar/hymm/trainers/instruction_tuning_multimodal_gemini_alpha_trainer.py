import gc
from functools import partial
import torch
from typing import Dict, Union

from index_kits.sampler import DistributedSampler
from torch.utils.data import DataLoader

from ..utils.torch_utils import set_worker_seed_builder, to_device
from ..data_kits.mmu_loader import MultiModalUnderstandingArrowStream
from ..data_kits.text_loader import TextArrowStream
from ..data_kits.t2i_loader import (
    TextImageArrowStream, TextImageToImageArrowStream, FaceIDArrowStream, TextImageInterleaveArrowStream
)
from ..trainers.multimodal_gemini_alpha_trainer import GeminiTrainerAlphaMultiModal
from ..data_kits.instruction_template import (
    text2image_instructions,
    inpainting_instructions,
    editing_instructions,
    subject_driven_instructions,
    face_id_instructions,
)

gc.set_threshold(7000, 100, 100)


class GeminiTrainerAlphaMultiModalInstruct(GeminiTrainerAlphaMultiModal):
    def build_dataloader(self):
        args = self.args

        self.dataloader_preliminary_setup()

        dataloader_kwargs = dict(
            **args.dataloader_params,
            worker_init_fn=set_worker_seed_builder(self.rank),
            shuffle=False,
            drop_last=True,
        )
        sampler_kwargs = dict(
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
        )

        def _filter_dummies(dummy_list):
            # Filter out dummies that are not in the dataset keys
            valid_dummies = []
            for dummy_candidate in dummy_list:
                if any(task in self.all_dataset_keys for task in self.dummy_to_tasks[dummy_candidate]):
                    valid_dummies.append(dummy_candidate)
            return valid_dummies

        # =====================================
        #     Text(+Image) to image data
        # =====================================
        self.task_dummy_dict['t2i'] = _filter_dummies(['mmu', 'face'])
        self.task_dummy_dict['face'] = _filter_dummies(['mmu'])
        task_info_list = [
            dict(dataset_tag="t2i", cur_task="t2i", cls=TextImageArrowStream, instruction_candidates=text2image_instructions),
            dict(dataset_tag="inpainting", cur_task="t2i", cls=TextImageToImageArrowStream, instruction_candidates=inpainting_instructions),
            dict(dataset_tag="editing_OmniEdit", cur_task="t2i", cls=TextImageToImageArrowStream, instruction_candidates=editing_instructions),
            dict(dataset_tag="editing_v2f", cur_task="t2i", cls=TextImageToImageArrowStream, instruction_candidates=editing_instructions),
            dict(dataset_tag="subject_driven", cur_task="t2i", cls=TextImageToImageArrowStream, instruction_candidates=subject_driven_instructions),
            dict(dataset_tag="face_id", cur_task="face", cls=FaceIDArrowStream, instruction_candidates=face_id_instructions),
            dict(dataset_tag="interleave", cur_task="t2i", cls=TextImageInterleaveArrowStream, instruction_candidates=None),
        ]
        for item in task_info_list:
            dataset_tag = item['dataset_tag']
            if dataset_tag not in self.all_dataset_keys:
                continue
            self.dataset_dict[dataset_tag] = item['cls'](
                args=args,
                dataset_tag=dataset_tag,
                tokenizer_name=args.tokenizer_name,
                task_kwargs=args.get(f'{dataset_tag}_task_kwargs'),
                index_kwargs=dict(
                    batch_size=self.micro_batch_size,  # Provide bsz to use multireso.
                    world_size=1,  # Dataset don't need to align with world_size. It will be handled by the sampler.
                    **args.get(f'{dataset_tag}_index_kwargs'),
                ),
                logger=self.logger,
                dummy_number=sum([self.dummy_dict[task] for task in self.task_dummy_dict[item['cur_task']]]),
                template="instruct",
                instruction_candidates=item['instruction_candidates'],
            )
            # Build sampler and data loader
            self.sampler_dict[dataset_tag] = DistributedSampler(
                self.dataset_dict[dataset_tag],
                num_replicas=self.dataset_num_replicas[dataset_tag],
                rank=self.dataset_rank[dataset_tag],
                batch_size=self.micro_batch_size,  # Provide bsz to use multireso.
                **sampler_kwargs,
            )
            self.dataloader_dict[dataset_tag] = DataLoader(
                self.dataset_dict[dataset_tag],
                batch_size=self.micro_batch_size,
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
        if "lm" in self.all_dataset_keys:
            lm_dataset_kwargs = dict(
                args=args,
                t2t_text_token_length=args.lm_token_length,
                tokenizer_name=args.tokenizer_name,
                index_kwargs=dict(
                    **args.lm_index_kwargs,
                ),
                logger=self.logger,
            )

            # ------------- LM ---------------
            self.dataset_dict['lm'] = TextArrowStream(
                index_file=args.lm_index_file,
                dataset_type="lm",
                template="instruct",
                **lm_dataset_kwargs,
            )
            self.sampler_dict['lm'] = DistributedSampler(
                self.dataset_dict['lm'],
                num_replicas=self.dataset_num_replicas['lm'],
                rank=self.dataset_rank['lm'],
                batch_size=self.micro_batch_size,
                **sampler_kwargs,
            )
            self.dataloader_dict['lm'] = DataLoader(
                self.dataset_dict['lm'],
                batch_size=self.micro_batch_size,
                sampler=self.sampler_dict['lm'],
                **dataloader_kwargs)

        # ================================================
        #          Multimodal understanding data
        # ================================================
        if "mmu" in self.all_dataset_keys:
            # Set the number of dummy tokens for mmu sequences
            self.task_dummy_dict['mmu'] = _filter_dummies(['t2i', 'face'])
            self.dataset_dict['mmu'] = MultiModalUnderstandingArrowStream(
                args=args,
                index_file=args.mmu_index_file,
                text_token_length=args.mmu_text_token_length,
                max_token_length=args.mmu_token_length,
                tokenizer_name=args.tokenizer_name,
                template="instruct",
                index_kwargs=dict(
                    **args.mmu_index_kwargs,
                ),
                dummy_number=sum([self.dummy_dict[task] for task in self.task_dummy_dict['mmu']]),
                logger=self.logger,
            )
            self.sampler_dict['mmu'] = DistributedSampler(
                self.dataset_dict['mmu'],
                num_replicas=self.dataset_num_replicas['mmu'],
                rank=self.dataset_rank['mmu'],
                **sampler_kwargs,
            )
            self.dataloader_dict['mmu'] = DataLoader(
                self.dataset_dict['mmu'],
                batch_size=self.micro_batch_size,
                sampler=self.sampler_dict['mmu'],
                **dataloader_kwargs,
            )

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        if batch["dtype"][0] == "t2i":
            inputs = self.prepare_model_t2i_inputs(batch, device, **kwargs)
        elif batch["dtype"][0] in ["inpainting", "editing_OmniEdit", "editing_v2f", "subject_driven", "interleave"]:
            inputs = self.prepare_model_ti2i_inputs(batch, device, **kwargs)
        elif batch["dtype"][0] == "face_id":
            inputs = self.prepare_model_face_inputs(batch, device, **kwargs)
        elif batch["dtype"][0] == "lm":
            inputs = self.prepare_model_lm_inputs(batch, device, **kwargs)
        elif batch["dtype"][0] == "mmu":
            inputs = self.prepare_model_mmu_inputs(batch, device, **kwargs)
        else:
            raise ValueError(f"Unknown batch dtype, expected {self.all_dataset_keys}, got {batch['dtype']}")
        return inputs

    def prepare_model_t2i_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,        # [b, seqlen]
            image_mask=image_mask,      # [b, seqlen]
            iw_ih_scatter_index=batch["iw_ih_scatter_index"].to(device),        # [b, 2]
            iw_ih_scatter_src=batch["iw_ih_scatter_src"].to(device),            # [b, 2]
            timestep_scatter_index=batch["timestep_scatter_index"].to(device),  # [b, 1]
        )
        for task in self.task_dummy_dict['t2i']:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )
        batch_size, n_tokens = tokens.shape

        # Attention mask
        attention_mask = batch["attention_mask"].to(device)

        # ===================================== prepare diffusion =====================================
        image = batch["image"].to(device)
        if kwargs.get('skip_vae_encode'):
            t, model_t, x_0, x_t, u_t = None, None, None, None, None
        else:
            t, model_t, x_0, x_t, u_t = self.vae_encode(image, sample_type="sample", n_tokens=n_tokens).values()

        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                         # [b, seqlen]
            x_t=x_t,                            # [b, c, h, w]
            t=model_t,                          # [b]
            diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
            target=target_tokens,               # [b, seqlen]
            attention_mask=attention_mask,      # [b, seqlen, seqlen]
            image_loss_weight=self.args.image_loss_weight,
            data_type=batch['data_type'][0],                    # For loss
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_ti2i_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)
        # For interleave data, src_image_mask and src_images can be None
        src_image_mask = batch["src_image_mask"][:, :-1].contiguous().to(device) \
            if batch["src_image_mask"] is not None else None

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,            # [b, seqlen]
            image_mask=image_mask,          # [b, seqlen]
            **(dict(src_image_mask=src_image_mask) if src_image_mask is not None else {}),  # [b, seqlen]
            iw_ih_scatter_index=to_device(batch["iw_ih_scatter_index"], device),        # [b, 2n]
            iw_ih_scatter_src=to_device(batch["iw_ih_scatter_src"], device),            # [b, 2n]
            timestep_scatter_index=to_device(batch["timestep_scatter_index"], device),  # [b, n]
        )
        for task in self.task_dummy_dict['t2i']:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )
        batch_size, n_tokens = tokens.shape

        # Attention mask
        attention_mask = batch["attention_mask"].to(device)

        # ===================================== prepare diffusion =====================================
        if kwargs.get('skip_vae_encode'):
            t, model_t, x_0, x_t, u_t = None, None, None, None, None
            input_src_t, input_src_x = None, None
        else:
            t, model_t, x_0, x_t, u_t = self.vae_encode(batch["image"], sample_type="sample").values()
            if batch["src_images"] is not None:
                _, input_src_t, _, input_src_x, _ = self.vae_encode(batch["src_images"], sample_type="sample_start").values()
            else:
                input_src_t, input_src_x = None, None

        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                         # [b, seqlen]
            x_t=x_t,                            # [b, c, h, w]
            t=model_t,                          # [b]
            diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
            src_x=input_src_x,                  # [b, c, h, w]
            src_t=input_src_t,                  # [b]
            target=target_tokens,               # [b, seqlen]
            attention_mask=attention_mask,      # [b, seqlen, seqlen]
            image_loss_weight=self.args.image_loss_weight,
            data_type=batch['data_type'][0],                    # For loss
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_face_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)
        # src_image_mask is used with x_t. Here we rename to und_image_mask.
        und_image_mask = batch["src_image_mask"][:, :-1].contiguous().to(device)
        src_face_embedding = batch["src_face_embedding"].to(device).unsqueeze(-1).unsqueeze(-1)     # [b, c, 1, 1]

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,                # [b, seqlen]
            image_mask=image_mask,              # [b, seqlen]
            und_image_masks=und_image_mask,     # [b, seqlen]
            iw_ih_scatter_index=batch["iw_ih_scatter_index"].to(device),        # [b, 2]
            iw_ih_scatter_src=batch["iw_ih_scatter_src"].to(device),            # [b, 2]
            timestep_scatter_index=batch["timestep_scatter_index"].to(device),  # [b, 1]
        )
        for task in self.task_dummy_dict['face']:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )
        batch_size, n_tokens = tokens.shape

        # Attention mask
        attention_mask = batch["attention_mask"].to(device)

        # ===================================== prepare diffusion =====================================
        image = batch["image"].to(device)
        t, model_t, x_0, x_t, u_t = self.vae_encode(image, sample_type="sample").values()

        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                             # [b, seqlen]
            x_t=x_t,                                # [b, c, h, w]
            t=model_t,                              # [b]
            diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
            target=target_tokens,                   # [b, seqlen]
            attention_mask=attention_mask,          # [b, seqlen, seqlen]
            image_loss_weight=self.args.image_loss_weight,
            data_type=batch["data_type"][0],        # For loss
            src_face_embedding=src_face_embedding,  # [b, c, 1, 1]
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_lm_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,
        )
        for task in self.task_dummy_dict['lm']:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )
        batch_size, n_tokens = tokens.shape

        # Attention mask
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
        attention_mask = causal_mask.view(1, 1, n_tokens, n_tokens).repeat(batch_size, 1, 1, 1)

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

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,        # [b, seqlen]
            und_image_masks=und_image_mask,  # [b, seqlen]
            iw_ih_scatter_index=batch["iw_ih_scatter_index"].to(device),        # [b, 2]
            iw_ih_scatter_src=batch["iw_ih_scatter_src"].to(device),            # [b, 2]
        )
        for task in self.task_dummy_dict['mmu']:
            if self.dummy_dict[task]:
                tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                    tokens, target_tokens, extra,
                    dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
                )
        batch_size, n_tokens = tokens.shape

        # build attention mask. Intra-image is full attention, inter-image and image-text is causal attention.
        attention_mask = batch["attention_mask"].to(device)

        images = batch["image"].to(device)
        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 512]
            target=target_tokens,  # [b, 512]
            attention_mask=attention_mask,  # [b, 512, 512]
            image_loss_weight=0,    # Set to zero to avoid dummy image tokens to affect the text loss
            data_type="mmu",        # For loss
            und_images=images,
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, batch_size, n_tokens
