import os
import einops
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from hymm.ar.mask_schedulers import create_attention_mask_general
from hymm.models.basic.rope import get_3d_rope
from hymm.diffusion import load_denoiser
from hymm.data_kits.arrow_dataset import ArrowDataset
from hymm.data_kits.datasets.csv_folder_dataset import CSVImageFolderDataset
from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.config import parse_eval_initial_args
from hymm.samplers.text2image_transfusion_sampler import Text2ImageTransfusionSampler
from hymm.utils.file_utils import rank0_logger
from hymm.utils.torch_utils import PRECISION_TO_TYPE


class InstructionTuningTransfusionSampler(Text2ImageTransfusionSampler):
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
            self.pil_image_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        elif self.vae_trans_type == "01":
            self.img_tensor_transform = transforms.Compose([])
            self.pil_image_to_tensor = transforms.Compose([transforms.ToTensor()])
        else:
            raise ValueError("Invalid trans_type: {}".format(self.vae_trans_type))

    # now we only support batch inference with the same src_img_token shape
    @torch.no_grad()
    def predict(self, task, instruction_list, src_img_tensor_batch, **kwargs):
        out_dict = {}

        guidance_scale = kwargs.get("guidance_scale", self.args.guidance_scale)
        default_face_guidance_scale = self.args.get("face_guidance_scale", 1.0)
        face_guidance_scale = kwargs.get("face_guidance_scale", default_face_guidance_scale)
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

        if src_img_tensor_batch.ndim == 4:
            src_img_tensor_list = []
            for src_img_tensor in src_img_tensor_batch:
                src_img_tensor_list.append(self.img_tensor_transform(src_img_tensor)[None])
            src_img_tensor = torch.cat(src_img_tensor_list, dim=0).to(self.device)
            n_srcs = 1
        elif src_img_tensor_batch.ndim == 5:
            src_img_tensor = src_img_tensor_batch.squeeze(0).to(self.device)
            n_srcs = len(src_img_tensor)
        else:
            raise ValueError(f"Invalid src_img_tensor_batch shape: {src_img_tensor_batch.shape}")

        h, w = src_img_tensor.shape[2], src_img_tensor.shape[3]
        patch_size = self.args.patch_size
        ds_factor = (self.vae_downsample_factor[0] * patch_size, self.vae_downsample_factor[1] * patch_size)
        
        cfg_factor = 1
        if guidance_scale > 1.0: 
            cfg_factor += 1
        if face_guidance_scale is not None and face_guidance_scale > 1.0:
            assert guidance_scale > 1.0, "Must conduct text guidance before face guidance"
            cfg_factor += 1
        self.logger.info(f'Set cfg_factor to {cfg_factor}, because guidance_scale={guidance_scale} and face_guidance_scale={face_guidance_scale}')

        assert h % ds_factor[0] == 0 and w % ds_factor[1] == 0, \
            f"Image size should be divisible by vae_downsample_factor * patch_size, but got ({h} x {w}) with vae_downsample_factor={self.vae_downsample_factor} and patch_size={patch_size}"

        th = h // ds_factor[0]
        tw = w // ds_factor[1]
        actual_image_token_length = th * tw

        image_token_shape_wh = torch.tensor([w, h, w, h], dtype=torch.long).unsqueeze(0).to(self.device)
        iw_ih_scatter_src = image_token_shape_wh.repeat(batch_size, 1)

        tokenizer = self.model_dict["tokenizer"]

        if task == "editing":
            tokens, iw_ih_scatter_index, _, text_mask, src_image_mask, tgt_image_mask = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion,
                prompt_list=instruction_list,
                infer_fn_kwargs_list=[dict(
                    image_token_length=actual_image_token_length,
                    src_image_token_lengths=[actual_image_token_length],
                    max_text_token_length=self.args.text_token_length + 1,
                    max_image_token_length=actual_image_token_length,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    add_timestep_token=self.args.add_timestep_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                ) for _ in range(batch_size)],
                do_classifier_free_guidance=guidance_scale > 1.0,
            )
        elif task == "inpainting":
            tokens, iw_ih_scatter_index, _, text_mask, src_image_mask, tgt_image_mask = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion,
                prompt_list=[(instruction, prompt) for instruction, prompt in zip(instruction_list, kwargs["prompt_list"])],
                infer_fn_kwargs_list=[dict(
                    image_token_length=actual_image_token_length,
                    src_image_token_lengths=[actual_image_token_length],
                    max_text_token_length=self.args.text_token_length + 1,
                    max_image_token_length=actual_image_token_length,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    add_timestep_token=self.args.add_timestep_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                ) for _ in range(batch_size)],
                do_classifier_free_guidance=guidance_scale > 1.0,
            )

            mask_batch = kwargs["mask_batch"].to(self.device)
            mask = torch.nn.functional.interpolate(mask_batch.unsqueeze(1), (h, w)) > 0.5
            src_img_tensor = src_img_tensor * (1 - mask.float())
        elif task == "subject_driven":
            tokens, iw_ih_scatter_index, _, text_mask, src_image_mask, tgt_image_mask = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion,
                prompt_list=instruction_list,
                infer_fn_kwargs_list=[dict(
                    image_token_length=th * tw,
                    src_image_token_lengths=[th * tw] * n_srcs,
                    max_text_token_length=self.args.text_token_length + 1,
                    max_total_token_length=self.args.text_token_length + 1 + self.args.image_token_length * 4,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    add_timestep_token=self.args.add_timestep_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                )],
                do_classifier_free_guidance=guidance_scale > 1.0,
            )
        elif task == "subject_driven2":     # for 3d RoPE and left padding
            tokens, iw_ih_scatter_index, text_mask, src_image_mask, tgt_image_mask = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion_editing2,
                prompt_list=instruction_list,
                infer_fn_kwargs_list=[dict(
                    src_image_token_lengths=[th * tw] * n_srcs,
                    tgt_image_token_length=th * tw,
                    max_text_token_length=self.args.text_token_length + 1,
                    max_total_token_length=self.args.text_token_length + 1 + self.args.image_token_length * 4,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    add_timestep_token=self.args.add_timestep_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                )],
                do_classifier_free_guidance=guidance_scale > 1.0,
            )
        elif task == "id":
            self.src_condition_type = self.args.get("src_condition_type", ["vae"])
            if isinstance(self.src_condition_type, str):
                self.src_condition_type = self.src_condition_type.split("_cat_")
            else: 
                assert isinstance(self.src_condition_type, list), f"src_condition_type should be a list,e.g, ['face_embed'], ['vae', 'face_embed'], but got {type(self.src_condition_type)}"

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
                self.args.get("face_bof_eof", False),
                self.args.get("resampler_token_length", None)
            )
            src_condition_lengths_dict["max_image_token_length_list"].append(actual_tgt_image_token_length)
            self.logger.info(f'src_condition_lengths_dict after prepare_src_condition_lengths: {src_condition_lengths_dict}')
            image_token_shape_wh = torch.tensor([w, h,]* (len(src_condition_lengths_dict["actual_src_image_token_length_list"]) +1 ), dtype=torch.long).unsqueeze(0).to(self.device)
            iw_ih_scatter_src = image_token_shape_wh.repeat(batch_size, 1)
            # reuse src_image_mask variable for src_face_embedding;
            tokens, iw_ih_scatter_index, _, text_mask, src_image_mask, tgt_image_mask = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_transfusion_faceid,
                prompt_list=instruction_list,
                infer_fn_kwargs_list=[dict(
                    image_token_length=         actual_tgt_image_token_length,
                    src_image_token_lengths=    src_condition_lengths_dict["actual_src_image_token_length_list"],
                    src_face_token_lengths=     src_condition_lengths_dict["actual_src_face_token_length_list"],
                    max_text_token_length=      self.args.text_token_length + 1,
                    max_image_token_length=     src_condition_lengths_dict["max_image_token_length_list"],
                    max_face_token_length=      src_condition_lengths_dict["max_face_token_length_list"],
                    add_iw_ih_token=        self.args.add_iw_ih_token,
                    add_timestep_token=     self.args.get('add_timestep_token', False),
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

        freqs_cos, freqs_sin = None, None
        if task == "subject_driven2":
            # build 3d RoPE
            assert len(tokens) == cfg_factor, "Only support batch size 1 for subject_driven task"
            start_pos = torch.where(tokens[0] == self.model_dict['tokenizer'].special_token_map["<img>"])[0][0].item()
            if self.use_3d_rope:
                img_pos = [start_pos + 3 * i for i in range(n_srcs + 1)]
                freqs_cos, freqs_sin = get_3d_rope(
                    self.args.rope_dim_list,
                    [h // ds_factor[0] for _ in range(n_srcs)] + [th],
                    [w // ds_factor[1] for _ in range(n_srcs)] + [tw],
                    max_len=self.args.text_token_length + 1 + n_srcs + (3 - n_srcs) * self.args.image_token_length,
                    img_pos=img_pos,
                    device=self.device,
                    theta=self.args.get('rope_theta', 10000),
                    use_real=True,
                    interleave=self.args.rope_type == "3d-interleave",
                    add_batch_axis=True,
                )
                if guidance_scale > 1.0:
                    freqs_cos = freqs_cos.repeat(2, 1, 1)
                    freqs_sin = freqs_sin.repeat(2, 1, 1)

            # build attention mask
            src_token_slices = []
            for i in range(len(src_img_tensor)):
                th_ = h // ds_factor[0]
                tw_ = w // ds_factor[1]
                src_token_slices.append(slice(start_pos, start_pos + th_ * tw_))
                start_pos += th_ * tw_ + 2
            tgt_token_slice = slice(start_pos, start_pos + actual_image_token_length)
            attention_mask = create_attention_mask_general(
                tokens,
                None,
                src_token_slices + [tgt_token_slice],
                mask_pad=False,
                return_inverse_mask=True,
                dtype=torch.float32,
            )[0]
        else:
            # build attention mask
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
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]

        if task != "id" or (task == "id" and "vae" in self.src_condition_type):
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                src_latents = self.model_dict["vae"].encode(src_img_tensor).latent_dist.sample()
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

        src_face_embedding = None
        if task == "id" and "face_embed" in self.src_condition_type:
            src_face_embedding = kwargs["src_face_embedding"].to(self.device)
            src_face_embedding = einops.rearrange(src_face_embedding, 'b c -> b c 1 1')

        # repeat for cfg
        if iw_ih_scatter_index is not None:
            iw_ih_scatter_src = iw_ih_scatter_src.repeat(cfg_factor, 1)
        if face_guidance_scale > 1.0:
            # if guidance_scale == 1.0,[pred_cond, pred_uncond_face] face_uncond_chunk_indices = 1
            # if guidance_scale >= 1.0,[pred_cond, pred_uncond_text, pred_uncond_face] face_uncond_chunk_indices = 2
            face_uncond_chunk_indices = 2 if guidance_scale > 1.0 else 1
        else:
            face_uncond_chunk_indices = None
        
        if src_x is not None:
            src_x = src_x.repeat(cfg_factor, 1, 1, 1)
            if face_uncond_chunk_indices is not None:
                src_x[face_uncond_chunk_indices, ...] = torch.zeros_like(src_x[face_uncond_chunk_indices, ...])
        
        if src_face_embedding is not None:
            src_face_embedding = src_face_embedding.repeat(cfg_factor, 1, 1, 1)
            if face_uncond_chunk_indices is not None:
                src_face_embedding[face_uncond_chunk_indices, ...] = torch.zeros_like(src_face_embedding[face_uncond_chunk_indices, ...])

        if src_model_t is not None:
            src_model_t = src_model_t.repeat(cfg_factor)

        if task in ["subject_driven", "subject_driven2"]:
            src_x = [src_x] * cfg_factor
            src_model_t = [src_model_t] * cfg_factor

        # with cond and uncond
        model_input_extra_kwargs = dict(
            idx=tokens,
            src_x=src_x,
            src_t=src_model_t,
            src_image_mask=src_image_mask,
            image_mask=tgt_image_mask,
            attention_mask=attention_mask,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )
        if iw_ih_scatter_index is not None:
            model_input_extra_kwargs.update({
                "iw_ih_scatter_index": iw_ih_scatter_index.to(self.device),
                "iw_ih_scatter_src": iw_ih_scatter_src.to(self.device),
            })
        if src_face_embedding is not None:
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

    sampler = InstructionTuningTransfusionSampler.from_pretrained(
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
        segments.append(f"face_cfg{args.face_guidance_scale}")
    dataset_kwargs = None

    if args.instruction_tuning_task == "editing":
        save_base = sampler.get_sample_save_dir(task=args.instruction_tuning_task, segments=segments)
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        dataset = ArrowDataset(
            arrow_file=args.editing_test_arrow,
            image_size=(args.sample_image_size, args.sample_image_size),
            length=1024,
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

    elif args.instruction_tuning_task == "inpainting":
        save_base = sampler.get_sample_save_dir(task=args.instruction_tuning_task, segments=segments)
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        dataset = ArrowDataset(
            arrow_file=args.inpainting_test_arrow,
            image_size=(args.sample_image_size, args.sample_image_size) if args.vae_type == "32x32-evaclip-sdxl" else None,
            length=1024,
            save_template=save_template,
            column_dict={"seed": "seed", "instruction": "instruction", "caption": "caption@json@long caption", "src_img": "src_img_bytes@bytes", "mask": "mask_token@ast"},
            subset="",
            seed_type="auto",
            seed=1234,
            seed_plus=0,
            skip_exist=False,
            logger=logger,
        )
        logger.info(f"Loaded {len(dataset)} data files in {args.inpainting_test_arrow}")
        input_batch_dict = {"instruction_list": "instruction", "prompt_list": "caption", "src_img_tensor_batch": "src_img", "mask_batch": "mask"}

    elif args.instruction_tuning_task in ["subject_driven", "subject_driven2"]:
        save_base = sampler.get_sample_save_dir(task=args.instruction_tuning_task, segments=segments)
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        dataset = args.csv
        ref_cols = ['ref1', 'ref2', 'ref3']

        def callback(item):
            src_img_tensors = []
            for obj in ref_cols:
                if Path(item[f'extra_{obj}']).exists():
                    src_img_tensors.append(sampler.pil_image_to_tensor(Image.open(item[f'extra_{obj}']).convert('RGB')))
                del item[f'extra_{obj}']
            item['src_img_tensor_batch'] = torch.stack(src_img_tensors)
            return item

        dataset_kwargs = {'extra_cols': ref_cols, 'callback': callback}
        input_batch_dict = {"instruction_list": "input", "src_img_tensor_batch": "src_img_tensor_batch"}
    elif args.instruction_tuning_task == "id":
        task = "id"
        save_base = sampler.get_sample_save_dir(task=task, segments=segments, testset= os.path.basename(args.id_test_saved_training_data))
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))


        dataset = CSVImageFolderDataset(
            source_dir=args.id_test_saved_training_data,
            save_template=save_template,
            image_size=(args.sample_image_size, args.sample_image_size),
            logger=logger,
            face_crop=args.get("face_crop", False),
            src_condition_type=args.get("src_condition_type", ["vae"]),
        )
        logger.info(f"Loaded {len(dataset)} data files in {args.id_test_saved_training_data}")
        input_batch_dict = {"instruction_list": "prompt", "src_img_tensor_batch": "src_image"}
        if args.get("src_condition_type", None) != None and "face_embed" in args.get("src_condition_type", None):
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

