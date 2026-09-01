
import os
import time
from pathlib import Path

import einops
import torch
import torchvision.transforms as transforms
from PIL import Image

from hymm.diffusion import load_denoiser
from hymm.data_kits.arrow_dataset import ArrowDataset
from hymm.data_kits.datasets.csv_folder_dataset import CSVImageFolderDataset
from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.config import parse_eval_initial_args
from hymm.samplers.text2image_transfusion_sampler import Text2ImageTransfusionSampler
from hymm.utils.file_utils import rank0_logger
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.data_kits.instruction_template import editing_instructions, subject_driven_instructions, face_id_instructions


class InstructionTuningGeminiAlphaSampler(Text2ImageTransfusionSampler):
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

        # only used to build src_x
        self.denoiser = load_denoiser(args)
        
        if self.vae_trans_type == "-11":
            self.img_tensor_transform = transforms.Compose(
                [
                    transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
                ]
            )
            # self.pil_image_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        elif self.vae_trans_type == "01":
            self.img_tensor_transform = transforms.Compose([])
            # self.pil_image_to_tensor = transforms.Compose([transforms.ToTensor()])
        else:
            raise ValueError("Invalid trans_type: {}".format(self.vae_trans_type))
    
    # list mode supports batch_size > 1
    # Now we assume src image(s) of every sample have the same shape, otherwise an error will be raised
    @torch.no_grad()
    def predict(self, task, instruction_list, src_img_tensor_batch=None, **kwargs):
        out_dict = {}

        guidance_scale = kwargs.get("guidance_scale", self.args.guidance_scale)
        face_guidance_scale = kwargs.get("face_guidance_scale", self.args.face_guidance_scale)
        diff_infer_steps = kwargs.get("diff_infer_steps", self.args.diff_infer_steps)
        output_type = kwargs.get("output_type", "pil")
        verbose = kwargs.get("verbose", 1)

        batch_size = len(instruction_list)
        seeds = self.prepare_seed(
            seed=kwargs.get('seed', None),
            batch_size=batch_size,
            num_sample_per_prompt=1,
        )

        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        if src_img_tensor_batch.ndim == 5:
            src_img_tensor_batch = [t for t in src_img_tensor_batch]


        # 在face_id_preserve训练时，如果是vae模式，src_img_tensor_batch是tensor，则会复用src_is_stacked_tensor为True时的逻辑（即inpainting、editing的逻辑），如果是face_embed模式，src_img_tensor_batch是list of None，
        # 但在face_id_preserve推理时，无论何种模式，src_img_tensor_batch都是tensor，因为需要根据src_img_tensor_batch确定target image的shape
        # 因此这里将face_id_preserve的处理逻辑独立出来单独处理，不像训练那样复用src_is_stacked_tensor为True时的逻辑
        special_task_list = ["face_id_preserve"]
        src_is_list_of_tensor = isinstance(src_img_tensor_batch, list) and task not in special_task_list
        src_is_stacked_tensor = isinstance(src_img_tensor_batch, torch.Tensor) and task not in special_task_list

        # ArrowDataset only apply ToTensor() transform, we need to apply remaining transforms corresponding to vae.trans_type
        if src_is_list_of_tensor: # batch_size * [n_srcs_{i} x c x h x w]
            src_img_tensor_list = []
            # n_srcs_src_img_tensor: n_srcs_{i} x c x h x w
            for n_srcs_src_img_tensor in src_img_tensor_batch:
                src_img_tensor_list_ = []
                # t: c x h x w
                for t in n_srcs_src_img_tensor:
                    src_img_tensor_list_.append(self.img_tensor_transform(t)[None])
                # each tensor is n_srcs_{i} x c x h x w, we assume all src images of every sample have the same shape, otherwise this will raise error
                src_img_tensor_list.append(torch.cat(src_img_tensor_list_, dim=0).to(self.device)) # batch_size * [n_srcs_{i} x c x h x w]
            n_srcs_list = [t.shape[0] for t in src_img_tensor_list] # batch_size
        
        elif src_is_stacked_tensor: # batch_size x c x h x w
            src_img_tensor_list = []
            for t in src_img_tensor_batch:
                src_img_tensor_list.append(self.img_tensor_transform(t)[None])
            src_img_tensor = torch.cat(src_img_tensor_list, dim=0).to(self.device) # batch_size x c x h x w
            n_srcs = 1
        
        elif task == "face_id_preserve":
            self.src_condition_type = self.args.get("face_id_preserve_src_condition_type", ["vae"])
            if isinstance(self.src_condition_type, str):
                self.src_condition_type = self.src_condition_type.split("_cat_")
            else: 
                assert isinstance(self.src_condition_type, list), f"src_condition_type should be a list,e.g, ['face_embed'], ['vae', 'face_embed'], but got {type(self.src_condition_type)}"

            self.face_bof_eof = self.args.get(f"face_id_preserve_face_bof_eof", False)
            self.resampler_token_length = self.args.get(f"face_id_preserve_resampler_token_length", None)
        
            src_img_tensor_list = []
            for t in src_img_tensor_batch:
                src_img_tensor_list.append(self.img_tensor_transform(t)[None])
            src_img_tensor = torch.cat(src_img_tensor_list, dim=0).to(self.device) # batch_size x c x h x w

        else:
            raise NotImplementedError()

        patch_size = self.args.patch_size
        ds_factor = (self.vae_downsample_factor[0] * patch_size, self.vae_downsample_factor[1] * patch_size)

        if src_is_list_of_tensor:
            th_list, tw_list, actual_image_token_length_list, iw_ih_scatter_src_list = [], [], [], []
            # t: n_srcs_{i} x c x h x w
            for t, n in zip(src_img_tensor_list, n_srcs_list):
                h, w = t.shape[2], t.shape[3]
                assert h % ds_factor[0] == 0 and w % ds_factor[1] == 0, f"Image size should be divisible by vae_downsample_factor * patch_size, but got ({h} x {w}) with vae_downsample_factor={self.vae_downsample_factor} and patch_size={patch_size}"

                th = h // ds_factor[0]
                tw = w // ds_factor[1]
                th_list.append(th)
                tw_list.append(tw)
                actual_image_token_length_list.append([th * tw] * n)

                # +1 for tgt image
                iw_ih_scatter_src_list.append(torch.tensor([w, h] * (n + 1), dtype=torch.long).unsqueeze(0).to(self.device))

        elif src_is_stacked_tensor:
            h, w = src_img_tensor.shape[2], src_img_tensor.shape[3]

            assert h % ds_factor[0] == 0 and w % ds_factor[1] == 0, f"Image size should be divisible by vae_downsample_factor * patch_size, but got ({h} x {w}) with vae_downsample_factor={self.vae_downsample_factor} and patch_size={patch_size}"

            th = h // ds_factor[0]
            tw = w // ds_factor[1]
            actual_image_token_length = th * tw

            # +1 for tgt image
            iw_ih_scatter_src = torch.tensor([w, h] * (n_srcs + 1), dtype=torch.long).unsqueeze(0).to(self.device)
            iw_ih_scatter_src = iw_ih_scatter_src.repeat(batch_size, 1)

        elif task == "face_id_preserve":
            h, w = src_img_tensor.shape[2], src_img_tensor.shape[3]
            th = h // ds_factor[0]
            tw = w // ds_factor[1]
            actual_image_token_length = th * tw


        cfg_factor = 1
        if guidance_scale > 1.0: 
            cfg_factor += 1
        if face_guidance_scale is not None and face_guidance_scale > 1.0:
            assert guidance_scale > 1.0, "Must conduct text guidance before face guidance"
            cfg_factor += 1

        tokenizer = self.model_dict["tokenizer"]

        if task=="inpainting_randomly_sampled_from_1024_high_quality_v2":
            tmp_instruction_list = []
            for system_prompt, content_prompt in zip(instruction_list, kwargs["prompt_list"]):
                tmp_instruction_list.append([
                    "User: ",
                    system_prompt.strip(),
                    " " + content_prompt.strip(),
                    "\n\n",
                    "Assistant: <answer>",
                ])
            instruction_list = tmp_instruction_list
            uncond_enabled = [
                False,
                True,
                True,
                False,
                False,
            ]

            tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, tgt_image_mask = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion,
                prompt_list=instruction_list,
                infer_fn_kwargs_list=[dict(
                    image_token_length=actual_image_token_length,
                    src_image_token_lengths=[actual_image_token_length],
                    max_text_token_length=self.args.text_token_length + 1,
                    max_image_token_length=actual_image_token_length,
                    uncond_enabled=uncond_enabled,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    add_timestep_token=self.args.add_timestep_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                ) for _ in range(batch_size)],
                do_classifier_free_guidance=guidance_scale > 1.0,
            )

            mask_batch = kwargs["mask_batch"].to(self.device)
            mask = torch.nn.functional.interpolate(mask_batch.unsqueeze(1), (h, w)) > 0.5
            src_img_tensor = src_img_tensor * (1 - mask.float())

        elif "editing" in task:
            tmp_instruction_list = []
            for instruction, seed in zip(instruction_list, seeds):
                system_prompt = editing_instructions[seed % len(editing_instructions)]
                tmp_instruction_list.append([
                    "User: ",
                    system_prompt.strip(),
                    " " + instruction.strip(),
                    "\n\n", 
                    "Assistant: <answer>",
                ])
            instruction_list = tmp_instruction_list
            uncond_enabled = [
                False,
                True,
                True,
                False,
                False,
            ]

            tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, tgt_image_mask = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion,
                prompt_list=instruction_list,
                infer_fn_kwargs_list=[dict(
                    image_token_length=actual_image_token_length,
                    src_image_token_lengths=[actual_image_token_length],
                    max_text_token_length=self.args.text_token_length + 1,
                    max_image_token_length=actual_image_token_length,
                    uncond_enabled=uncond_enabled,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    add_timestep_token=self.args.add_timestep_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                ) for _ in range(batch_size)],
                do_classifier_free_guidance=guidance_scale > 1.0,
            )

        elif task=="subject_driven":
            tmp_instruction_list = []
            for content_prompt, seed in zip(instruction_list, seeds):
                system_prompt = subject_driven_instructions[seed % len(subject_driven_instructions)]
                tmp_instruction_list.append([
                    "User: ",
                    system_prompt.strip(),
                    " " + content_prompt.strip(),
                    "\n\n",
                    "Assistant: <answer>",
                ])
            instruction_list = tmp_instruction_list
            uncond_enabled = [
                False,
                True,
                True,
                False,
                False,
            ]

            tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, tgt_image_mask = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion,
                prompt_list=instruction_list,
                infer_fn_kwargs_list=[dict(
                    image_token_length=th_list[k] * tw_list[k],
                    src_image_token_lengths=actual_image_token_length_list[k],
                    max_text_token_length=self.args.text_token_length + 1,
                    max_total_token_length=self.args.text_token_length + 1 + self.args.image_token_length * 4,
                    uncond_enabled=uncond_enabled,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    add_timestep_token=self.args.add_timestep_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                ) for k in range(batch_size)],
                do_classifier_free_guidance=guidance_scale > 1.0,
            )

        elif task=="face_id_preserve":
            tmp_instruction_list = []
            for content_prompt, seed in zip(instruction_list, seeds):
                system_prompt = face_id_instructions[seed % len(face_id_instructions)]
                tmp_instruction_list.append([
                    "User: ",
                    system_prompt.strip(),
                    " " + content_prompt.strip(),
                    "\n\n",
                    "Assistant: <answer>",
                ])
            instruction_list = tmp_instruction_list
            uncond_enabled = [
                False,
                True,
                True,
                False,
                False,
            ]

            actual_tgt_image_token_length = actual_image_token_length
            actual_src_image_token_length_vae = actual_image_token_length
            
            if "clip" in self.src_condition_type:
                # (TODO) face generation conditioned on clip embedding is not supported yet
                actual_src_image_token_length_clip = tokenizer.get_actual_image_token_length(src_img_tensor, self.clip_meta_info, patch_size=self.args.get("patch_size_clip", 1))
            else:
                actual_src_image_token_length_clip = None

            src_condition_lengths_dict = tokenizer.prepare_src_condition_lengths(
                actual_src_image_token_length_vae,
                actual_src_image_token_length_vae,
                actual_src_image_token_length_clip,
                actual_src_image_token_length_clip,
                self.src_condition_type,
                self.face_bof_eof,
                self.resampler_token_length,
            )
            src_condition_lengths_dict["max_image_token_length_list"].append(actual_tgt_image_token_length)

            image_token_shape_wh = torch.tensor([w, h,]* (len(src_condition_lengths_dict["actual_src_image_token_length_list"]) +1 ), dtype=torch.long).unsqueeze(0).to(self.device)
            iw_ih_scatter_src = image_token_shape_wh.repeat(batch_size, 1)
            # reuse src_image_mask variable for src_face_embedding;
            tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, tgt_image_mask = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion_faceid,
                prompt_list=instruction_list,
                infer_fn_kwargs_list=[dict(
                    image_token_length=actual_tgt_image_token_length,
                    src_image_token_lengths=src_condition_lengths_dict["actual_src_image_token_length_list"],
                    src_face_token_lengths=src_condition_lengths_dict["actual_src_face_token_length_list"],
                    max_text_token_length=self.args.text_token_length + 1,
                    max_image_token_length=src_condition_lengths_dict["max_image_token_length_list"],
                    max_face_token_length=src_condition_lengths_dict["max_face_token_length_list"],
                    uncond_enabled=uncond_enabled,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    add_timestep_token=self.args.add_timestep_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                ) for _ in range(batch_size)],
                do_classifier_free_guidance=guidance_scale > 1.0,
                condition_repeat_times=1,
                uncondition_repeat_times=(cfg_factor-1), # for text drop in text and face guidance
            )
            # Return the save_src_img_tensor in range [0, 1] for visualization
            out_dict["save_src_img_tensor"] = ( src_img_tensor + 1.0 ) * 0.5
            
        else:
            raise NotImplementedError(f"Not implemented task: {task}")

        tokens = tokens[:, :-1].contiguous().to(self.device)
        src_image_mask = src_image_mask[:, :-1].contiguous().to(self.device)
        tgt_image_mask = tgt_image_mask[:, :-1].contiguous().to(self.device)

        n_tokens = tokens.shape[1]
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=self.device).tril(diagonal=0)
        causal_mask = causal_mask.view(1, n_tokens, n_tokens).repeat(tokens.shape[0], 1, 1)
        image_mask_1 = src_image_mask.view(tokens.shape[0], 1, n_tokens).repeat(1, n_tokens, 1)
        image_mask_2 = image_mask_1.transpose(1, 2)
        image_mask_3 = tgt_image_mask.view(tokens.shape[0], 1, n_tokens).repeat(1, n_tokens, 1)
        image_mask_4 = image_mask_3.transpose(1, 2)
        attention_mask = causal_mask | (image_mask_1.bool() & image_mask_2.bool()) | (image_mask_3.bool() & image_mask_4.bool())
        # unsqueeze for attention head dim
        attention_mask = attention_mask.unsqueeze(1)

        # ===================================== prepare diffusion =====================================
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        if src_is_list_of_tensor:
            batch_src_xs = []
            batch_src_model_ts = []
            for src_images in src_img_tensor_list:
                src_xs = []
                src_model_ts = []
                for src_image in src_images: # c x h x w
                    with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                        # c x h x w -> 1 x c x h x w -> 1 x c x 1 x h x w
                        src_latents = self.model_dict["vae"].encode(src_image[None]).latent_dist.sample()
                        if hasattr(self.model_dict["vae"].config, 'shift_factor') and self.model_dict["vae"].config.shift_factor:
                            src_latents.sub_(self.model_dict["vae"].config.shift_factor).mul_(self.model_dict["vae"].config.scaling_factor)
                        else:
                            src_latents.mul_(self.model_dict["vae"].config.scaling_factor)

                    if hasattr(self.model_dict["vae"], "ffactor_temporal") and self.model_dict["vae"].ffactor_temporal == 1:
                        assert src_latents.shape[2] == 1, "src_latents should have shape [B, C, T, H, W] and T should be 1"
                        # 1 x c x h x w
                        src_latents = src_latents.squeeze(2)

                    src_t, src_x_0, src_x_1 = self.denoiser.sample_start(src_latents)
                    src_t, src_x, _ = self.denoiser.path_sampler.plan(src_t, src_x_0, src_x_1)
                    src_model_t = self.denoiser.get_model_t(src_t)  # t*1000
                    src_xs.append(src_x)
                    src_model_ts.append(src_model_t)
                
                # batch_src_xs is a list of tensors, the length of the list is batch_size.
                # each tensor is n_src_{i} x c x h x w, n_src_{i} is the number of source images of the i-th sample.
                batch_src_xs.append(torch.cat(src_xs))
                # batch_src_model_ts is a list of tensors, the length of the list is batch_size.
                # each tensor is n_src_{i}, n_src_{i} is the number of source images of the i-th sample.
                batch_src_model_ts.append(torch.cat(src_model_ts))

                input_src_x = batch_src_xs
                input_src_t = batch_src_model_ts
        elif src_is_stacked_tensor:
            src_images = src_img_tensor
            with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                src_latents = self.model_dict["vae"].encode(src_images).latent_dist.sample()
                if hasattr(self.model_dict["vae"].config, 'shift_factor') and self.model_dict["vae"].config.shift_factor:
                    src_latents.sub_(self.model_dict["vae"].config.shift_factor).mul_(self.model_dict["vae"].config.scaling_factor)
                else:
                    src_latents.mul_(self.model_dict["vae"].config.scaling_factor)

            if hasattr(self.model_dict["vae"], "ffactor_temporal") and self.model_dict["vae"].ffactor_temporal == 1:
                assert src_latents.shape[2] == 1, "src_latents should have shape [B, C, T, H, W] and T should be 1"
                src_latents = src_latents.squeeze(2)

            src_t, src_x_0, src_x_1 = self.denoiser.sample_start(src_latents)
            src_t, src_x, _ = self.denoiser.path_sampler.plan(src_t, src_x_0, src_x_1)
            src_model_t = self.denoiser.get_model_t(src_t) # t*1000

            input_src_x = src_x
            input_src_t = src_model_t
        
        elif task == "face_id_preserve":
            if "vae" in self.src_condition_type:
                src_images = src_img_tensor
                with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                    src_latents = self.model_dict["vae"].encode(src_images).latent_dist.sample()
                    if hasattr(self.model_dict["vae"].config, 'shift_factor') and self.model_dict["vae"].config.shift_factor:
                        src_latents.sub_(self.model_dict["vae"].config.shift_factor).mul_(self.model_dict["vae"].config.scaling_factor)
                    else:
                        src_latents.mul_(self.model_dict["vae"].config.scaling_factor)
                src_t, src_x_0, src_x_1 = self.denoiser.sample_start(src_latents)
                src_t, src_x, _ = self.denoiser.path_sampler.plan(src_t, src_x_0, src_x_1)
                src_model_t = self.denoiser.get_model_t(src_t)  # t*1000
            else:
                src_x = None
                src_model_t = None

            input_src_x = src_x
            input_src_t = src_model_t

            src_face_embedding = None
            if "face_embed" in self.src_condition_type:
                src_face_embedding = kwargs["src_face_embedding"].to(self.device)
                src_face_embedding = einops.rearrange(src_face_embedding, 'b c -> b c 1 1')


        if guidance_scale > 1.0:
            if src_is_list_of_tensor:
                input_src_x = input_src_x * cfg_factor
                input_src_t = input_src_t * cfg_factor
                input_iw_ih_scatter_src = iw_ih_scatter_src_list * cfg_factor
                input_iw_ih_scatter_index = [t.cuda() for t in iw_ih_scatter_index]
                input_timestep_scatter_index = [t.cuda() for t in timestep_scatter_index]

            elif src_is_stacked_tensor:
                input_src_x = src_x.repeat(cfg_factor, 1, 1, 1)
                input_src_t = src_model_t.repeat(cfg_factor)
                input_iw_ih_scatter_src = iw_ih_scatter_src.repeat(cfg_factor, 1)
                input_iw_ih_scatter_index = iw_ih_scatter_index.cuda()
                input_timestep_scatter_index = timestep_scatter_index.cuda()
            
            elif task == "face_id_preserve":
                if face_guidance_scale > 1.0:
                # if guidance_scale == 1.0,[pred_cond, pred_uncond_face] face_uncond_chunk_indices = 1
                # if guidance_scale >= 1.0,[pred_cond, pred_uncond_text, pred_uncond_face] face_uncond_chunk_indices = 2
                    face_uncond_chunk_indices = 2 if guidance_scale > 1.0 else 1
                else:
                    face_uncond_chunk_indices = None
                
                if input_src_x is not None:
                    input_src_x = input_src_x.repeat(cfg_factor, 1, 1, 1)
                    if face_uncond_chunk_indices is not None:
                        input_src_x[face_uncond_chunk_indices, ...] = torch.zeros_like(src_x[face_uncond_chunk_indices, ...])
                if src_model_t is not None:
                    src_model_t = src_model_t.repeat(cfg_factor)
                
                if src_face_embedding is not None:
                    src_face_embedding = src_face_embedding.repeat(cfg_factor, 1, 1, 1)
                    if face_uncond_chunk_indices is not None:
                        src_face_embedding[face_uncond_chunk_indices, ...] = torch.zeros_like(src_face_embedding[face_uncond_chunk_indices, ...])

                input_iw_ih_scatter_src = iw_ih_scatter_src.repeat(cfg_factor, 1)
                input_iw_ih_scatter_index = iw_ih_scatter_index.cuda()
                input_timestep_scatter_index = timestep_scatter_index.cuda()


        # with cond and uncond
        model_input_extra_kwargs = dict(
            idx=tokens,
            src_x=input_src_x,
            src_t=input_src_t,
            src_image_mask=src_image_mask,
            image_mask=tgt_image_mask,
            attention_mask=attention_mask,
            freqs_cos=None,
            freqs_sin=None,
        )
        if iw_ih_scatter_index is not None:
            model_input_extra_kwargs.update({
                "iw_ih_scatter_index": input_iw_ih_scatter_index,
                "iw_ih_scatter_src": input_iw_ih_scatter_src,
            })
        if timestep_scatter_index is not None:
            model_input_extra_kwargs.update({
                "timestep_scatter_index": input_timestep_scatter_index,
            })
        if task == "face_id_preserve" and src_face_embedding is not None:
            model_input_extra_kwargs.update({
                "src_face_embedding": src_face_embedding.to(self.device),
            })


        if verbose == 1:
            info_str = f"""
            prompt: {instruction_list}
            """
            info_str += f"""
            image_size: {(h, w)}
                  seed: {seeds}
      diff_infer_steps: {diff_infer_steps}
        guidance_scale: {guidance_scale}
            """
            self.logger.info(info_str)

        start_time = time.time()
        samples = self.pipeline(batch_size=batch_size,
                                image_size=(h, w),
                                num_inference_steps=diff_infer_steps,
                                guidance_scale=guidance_scale,
                                face_guidance_scale=face_guidance_scale,
                                generator=generators,
                                output_type=output_type,
                                model_input_extra_kwargs=model_input_extra_kwargs,
                                )[0]
        out_dict['samples'] = samples
        gen_time = time.time() - start_time
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")

        return out_dict


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)

    sampler = InstructionTuningGeminiAlphaSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args

    # Start evaluation
    segments = [f"{args.denoise_type}{args.diff_infer_steps}", f"cfg{args.guidance_scale}"]
    if args.get("face_guidance_scale", None) is not None:
        segments.append(f"facecfg{args.face_guidance_scale}")
    dataset_kwargs = None


    if args.instruction_tuning_task == "inpainting_randomly_sampled_from_1024_high_quality_v2":
        save_base = sampler.get_sample_save_dir(task=args.instruction_tuning_task, segments=segments)
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        if "flux" in args.inpainting_randomly_sampled_from_1024_high_quality_v2_test_arrow:
            column_dict = {
                "seed": "seed",
                "instruction": "instruction",
                "caption": "caption",
                "src_img": "image_bytes@bytes",
                "mask": "mask_bytes@bytes",
            }
            image_size = args.get("editing_sample_image_size", 1024)
        else:
            column_dict = {
                "seed": "seed",
                "instruction": "instruction",
                "caption": "caption_v2@json@long_caption",
                "src_img": "src_img_bytes@bytes",
                "mask": "mask@ast",
            }
            image_size = None
        dataset = ArrowDataset(
            arrow_file=args.inpainting_randomly_sampled_from_1024_high_quality_v2_test_arrow,
            image_size=image_size,
            save_template=save_template,
            # length=4,
            # column_dict={"seed": "seed", "instruction": "instruction", "caption": "caption", "src_img": "src_img_bytes", "mask": "mask"}, # inpainting_randomly_sampled_from_237.5M
            column_dict=column_dict, # inpainting_randomly_sampled_from_1024_high_quality_v2
            subset="",
            seed_type="auto",
            seed=1234,
            seed_plus=0,
            skip_exist=False,
            logger=logger,
        )

        input_batch_dict = {"instruction_list": "instruction", "prompt_list": "caption", "src_img_tensor_batch": "src_img", "mask_batch": "mask"}

    elif args.instruction_tuning_task == "editing_video2frame_in_house":
        save_base = sampler.get_sample_save_dir(task=args.instruction_tuning_task, segments=segments)
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        dataset = ArrowDataset(
            arrow_file=args.editing_video2frame_in_house_test_arrow,
            image_size=None,
            save_template=save_template,
            column_dict={"seed": "seed", "instruction": "instruction", "src_img": "src_img_bytes@bytes"},
            subset="",
            seed_type="auto",
            seed=1234,
            seed_plus=0,
            skip_exist=False,
            logger=logger,
        )

        input_batch_dict = {"instruction_list": "instruction", "src_img_tensor_batch": "src_img"}

    elif "editing" in args.instruction_tuning_task:
        save_base = sampler.get_sample_save_dir(task=args.instruction_tuning_task, segments=segments)
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        
        if hasattr(args, "editing_test_arrow") and "ImagenHub" in args.editing_test_arrow:
            arrow_file = args.editing_test_arrow
            image_size = args.get("editing_sample_image_size", 1024)
            column_dict={"seed": "seed", "instruction": "instruction",              "src_img": "src_img_bytes@bytes"}
        else:
            arrow_file = args.editing_OmniEdit_test_arrow
            image_size = None
            column_dict={"seed": "seed", "instruction": "instruction", "src_img": "src_img_bytes@bytes"}

        dataset = ArrowDataset(
            arrow_file=arrow_file,
            image_size=image_size,
            save_template=save_template,
            column_dict=column_dict,
            subset="",
            seed_type="auto",
            seed=1234,
            seed_plus=0,
            skip_exist=False,
            logger=logger,
        )

        input_batch_dict = {"instruction_list": "instruction", "src_img_tensor_batch": "src_img"}


    elif args.instruction_tuning_task == "subject_driven":
        save_base = sampler.get_sample_save_dir(task=args.instruction_tuning_task, segments=segments)
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        dataset = args.csv
        
        if "dreambooth" in args.csv:
            ref_cols = ['ref1']
        elif "subject_driven_two_subject_ImageHub" in args.csv:
            ref_cols = ['ref1', 'ref2']
        else:
            ref_cols = ['ref1', 'ref2', 'ref3']

        def callback(item):
            src_img_tensors = []
            for obj in ref_cols:
                if Path(item[f'extra_{obj}']).exists():
                    # follow ArrowDataset, we only apply ToTensor() transform
                    src_img_tensors.append(
                        transforms.ToTensor()(Image.open(item[f'extra_{obj}']).convert('RGB'))
                    )
                del item[f'extra_{obj}']
            item['src_img_tensor_batch'] = torch.stack(src_img_tensors)  # n_srcs x c x h x w
            return item

        dataset_kwargs = {'extra_cols': ref_cols, 'callback': callback}
        input_batch_dict = {"instruction_list": "input", "src_img_tensor_batch": "src_img_tensor_batch"}

    elif args.instruction_tuning_task == "face_id_preserve":
        save_base = sampler.get_sample_save_dir(task=args.instruction_tuning_task, segments=segments, testset=os.path.basename(args.face_id_preserve_test_dir))
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))


        dataset = CSVImageFolderDataset(
            source_dir=args.face_id_preserve_test_dir,
            save_template=save_template,
            image_size=(args.sample_image_size, args.sample_image_size),
            logger=logger,
            face_crop=args.get("face_id_preserve_face_crop", False),
            src_condition_type=args.get("face_id_preserve_src_condition_type", "vae"),
        )
        logger.info(f"Loaded {len(dataset)} data files in {args.face_id_preserve_test_dir}")
        input_batch_dict = {"instruction_list": "prompt", "src_img_tensor_batch": "src_image"}
        if args.get("face_id_preserve_src_condition_type", None) != None and "face_embed" in args.get("face_id_preserve_src_condition_type", None):
            input_batch_dict["src_face_embedding"] = "src_face_embedding"

    else:
        raise NotImplementedError(f"Not implemented instruction tuning task: {args.instruction_tuning_task}")

    dataloader = sampler.build_sample_dataloader(
        data_source=dataset,
        save_template=save_template,
        batch_size=args.sample_batch_size,
        dataset_kwargs=dataset_kwargs,
    )

    sampler.batch_sample(
        dataloader=dataloader,
        input_batch_dict=input_batch_dict,
        # other kwargs passed to predict
        task=args.instruction_tuning_task,
    )

    logger.info(f"Save the generated images to: {save_base}")


if __name__ == "__main__":
    main()
