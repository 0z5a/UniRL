from einops import rearrange

import torch

from hymm.data_kits.arrow_dataset import ArrowDataset
from hymm.samplers.base_sampler import setup_distributed_initialize
from hymm.config import parse_eval_initial_args
from hymm.samplers.text2image_ar_sampler import Text2ImageARSampler
from hymm.utils.file_utils import rank0_logger


class InstructionTuningARSampler(Text2ImageARSampler):
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
    
    # now we only support batch inference with the same src_img_token shape
    @torch.no_grad()
    def predict(self, task, instruction_list, src_img_token_batch, **kwargs):
        batch_size = len(instruction_list)
        seeds = self.prepare_seed(
            seed=kwargs.get('seed', None),
            batch_size=batch_size,
            num_sample_per_prompt=1,
        )

        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        latent_h = src_img_token_batch.shape[1]
        latent_w = src_img_token_batch.shape[2]
        image_seq_len = latent_h * latent_w

        tokenizer = self.model_dict["tokenizer"]

        if task == "editing":
            input_ids, iw_ih_scatter_index, batched_real_pos = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_ar_editing_infer,
                prompt_list=instruction_list,
                infer_fn_kwargs_list=[dict(
                    src_img_token=src_img_token_batch[i].reshape(-1) + self.image_token_offset,
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                ) for i in range(batch_size)],
                do_classifier_free_guidance=self.args.guidance_scale > 1.0,
            )
        elif task == "inpainting":
            input_ids, iw_ih_scatter_index, batched_real_pos = tokenizer.batch_gen_infer(
                infer_fn=tokenizer.encode_ar_inpainting_infer,
                prompt_list=list(zip(instruction_list, kwargs["prompt_list"])),
                infer_fn_kwargs_list=[dict(
                    uncond_enabled=[False, True],     # [instruction, prompt]: only apply uncond on prompt
                    src_img_token=src_img_token_batch[i].reshape(-1) + self.image_token_offset,
                    mask=kwargs["mask_batch"][i],
                    add_iw_ih_token=self.args.add_iw_ih_token,
                    use_front_boi_token=self.args.use_front_boi_token,
                ) for i in range(batch_size)],
                do_classifier_free_guidance=self.args.guidance_scale > 1.0,
            )
        else:
            raise NotImplementedError(f"Not implemented instruction tuning task: {task}")

        input_ids = input_ids.to(self.device)
        batched_real_pos = batched_real_pos.to(self.device)

        iw_ih_scatter_src = None
        if iw_ih_scatter_index is not None:
            iw_ih_scatter_index = iw_ih_scatter_index.to(self.device)
            iw_ih_scatter_src = torch.tensor([latent_w, latent_h, latent_w, latent_h], dtype=torch.long, device=self.device)
            iw_ih_scatter_src = iw_ih_scatter_src.unsqueeze(0).repeat(input_ids.shape[0], 1)

        model_intput_kwargs = {
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": iw_ih_scatter_src,
        }

        image_token_ids = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=self.device)
        input_pos = torch.arange(0, input_ids.shape[1], device=self.device, dtype=torch.long)

        current_pos = batched_real_pos
        if task == "inpainting" and self.args.inpainting_blend: # only support batch_size == 1
            src_img_token = src_img_token_batch[0].reshape(-1).cuda()
            mask = kwargs["mask_batch"][0].reshape(-1).cuda()
        self.model_dict["model"].set_kv_cache(batch_size=input_ids.shape[0], device=self.device)
        for i in range(image_seq_len):
            img_token = self.get_next_token(
                input_ids,
                input_pos,
                generators=generators,
                no_iw_ih_scatter=(i!=0),
                real_pos=current_pos-1,
                **model_intput_kwargs
            )
            if task == "inpainting" and self.args.inpainting_blend: # only support batch_size == 1
                if mask[i] == 0.0:
                    img_token = src_img_token[i:i+1].unsqueeze(-1)
                    img_token = img_token.expand(2, 1)
            next_token = img_token + self.image_token_offset
            image_token_ids = torch.cat([image_token_ids, img_token], dim=-1)
            # print(image_token_ids)
            input_ids = next_token
            # input_pos = torch.tensor([current_pos], device=self.device, dtype=torch.long)
            input_pos = current_pos.clone()
            current_pos += 1
        # must clear kv cache in interactive mode
        self.model_dict["model"].clear_kv_cache()
        if self.cfg_enabled:
            image_token_ids, _ = torch.chunk(image_token_ids, chunks=2, dim=0)
        image_token_ids = rearrange(image_token_ids, "b (h w) -> b h w", h=latent_h, w=latent_w)
        with torch.autocast(device_type="cuda", dtype=self.vae_dtype, enabled=self.vae_dtype != torch.float32):
            results_img = self.model_dict["vae"].vq_decode(image_token_ids)
        if self.model_dict["vae"]._trans_type == "-11":
            results_img = results_img.clamp(-1, 1) * 0.5 + 0.5
        elif self.model_dict["vae"]._trans_type == "01":
            results_img = results_img.clamp(0, 1)
        return {"samples": results_img}


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    logger = rank0_logger(rank)

    sampler = InstructionTuningARSampler.from_pretrained(
        ckpt_path=initial_args.ckpt,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    # Get updated args (include the yaml configs saved along with model checkpoint)
    args = sampler.args

    # Start evaluation
    # add logits processor config into segments automatically, cfg is included
    segments = []
    logits_processor_list = args.get("logits_processors_cfg", [])
    for logits_processor in logits_processor_list:
        for name, kwargs in logits_processor.items():
            if name in ["CfgLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"]:
                for k, v in kwargs.items():
                    seg = f"{k}{v}"
                    segments.append(seg)

    if args.instruction_tuning_task == "editing":
        task="editing"
        save_base = sampler.get_sample_save_dir(task=task, segments=segments)
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        dataset = ArrowDataset(
            arrow_file=args.editing_test_arrow,
            length=1024,
            save_template=save_template,
            column_dict={"seed": "seed", "instruction": "instruction", "src_img_token": "src_img_token_256x256_88-vqgan-hy_241024@ast"},
            subset="",
            seed_type="auto",
            seed=1234,
            seed_plus=0,
            skip_exist=False,
            logger=logger,
        )

        input_batch_dict={"instruction_list": "instruction", "src_img_token_batch": "src_img_token"}

    elif args.instruction_tuning_task == "inpainting":
        task="inpainting"
        save_base = sampler.get_sample_save_dir(task=task, segments=segments)
        logger.info(f"Save the generated images to: {save_base}")
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        dataset = ArrowDataset(
            arrow_file=args.inpainting_test_arrow,
            length=1024,
            save_template=save_template,
            column_dict={"seed": "seed", "instruction": "instruction", "caption":"caption@json@long caption", "src_img_token": "img_token_88-vqgan-hy_241024@ast", "mask": "mask_token@ast"},
            subset="",
            seed_type="auto",
            seed=1234,
            seed_plus=0,
            skip_exist=False,
            logger=logger,
        )

        input_batch_dict={"instruction_list": "instruction", "prompt_list": "caption", "src_img_token_batch": "src_img_token", "mask_batch": "mask"}

    else:
        raise NotImplementedError(f"Not implemented instruction tuning task: {args.instruction_tuning_task}")

    dataloader = sampler.build_sample_dataloader(
        data_source=dataset,
        batch_size=args.sample_batch_size,
    )

    sampler.batch_sample(
        dataloader=dataloader,
        input_batch_dict=input_batch_dict,
        # other kwargs passed to predict
        task=task,
    )

    logger.info(f"Save the generated images to: {save_base}")


if __name__ == "__main__":
    main()
