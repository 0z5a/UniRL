import copy
import os
from functools import partial
import threading
import subprocess
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from hymm.data_kits.av_loader import MultimodalAVIndexDataset
from hymm.data_kits.utils.data_container import MultimodalDataContainer
from hymm.data_kits.csv_dataset import MessageListDataset
from hymm.data_kits.datasampler import DistributedSamplerFix

from hymm.models import build_model
from hymm.models.tokenizers import load_tokenizer
from hymm.models.diffusion.leo import MOE_LAYER_IMPL
from hymm.trainers.pretrain_pure_torch import find_engine
from hymm.core.global_vars import get_denoiser, get_mm_state, get_combined_iterator
from hymm.core.data_provider import DatasetsProvider
from hymm.core.data_provider_dit import prepare_model_t2v_inputs
from hymm.utils.helpers import readable_time
from hymm.utils.torch_utils import Timer

from hymm.trainers.pretrain_pure_torch_dit import Leo2Trainer

import warnings
# Ignore the distributed device fallback warning
warnings.filterwarnings(
    "ignore", 
    message="No device id is provided via `init_process_group` or `barrier `"
)

class CSVIndexManager:
    def __init__(self, required_cols, index_kwargs, logger):
        self.required_cols = required_cols
        self.index_kwargs = index_kwargs
        self.logger = logger
        self.datas = []

        for csv_path in self.index_kwargs['index_file']:
            df = pd.read_csv(csv_path)
            df = df[required_cols]
            for i, row in df.iterrows():
                self.datas.append(dict(zip(required_cols, row)))
        self.indices = list(range(len(self.datas)))
        self.logger.info(f'Load samples num = {len(self.datas)}') 

    def __len__(self):
        return len(self.datas)

    def shuffle(self, seed, fast=True, **kwargs):         
        random.seed(seed)
        random.shuffle(self.datas)

    def get_attribute(self, idx, column_name):
        return self.datas[idx][column_name]

class VideoDPOLatentDataset(MultimodalAVIndexDataset):

    def collate_fn(self, batch):
        result = {}
        for key in batch[0].keys():
            if key in ("rope_pair", "dummy_type_dict"):
                result[key] = [item[key] for item in batch]
            else:
                result[key] = default_collate([item[key] for item in batch])
        return result

    def setup_index_manager(self, batch_size):
        required_cols = ['task_type', 'win_source','lose_source','win_video_path', 'win_vae_cache_path', 'lose_video_path', 'lose_vae_cache_path', 'prompt']
        self.index_manager = CSVIndexManager(required_cols, self.index_kwargs, self.logger)

    def get_batch(self, idx):
        args = self.args
        item = self.index_manager.datas[idx]
        task_type = item['task_type']

        video_pair = []
        audio_pair = []
        rope_pair = []

        for i in range(2):
            if i == 0:
                vae_cache_path = item['win_vae_cache_path']
                video_path_col = 'win_video_path'
            else:
                vae_cache_path = item['lose_vae_cache_path']
                video_path_col = 'lose_video_path'

            # Get latents
            latents_np = np.load(vae_cache_path).squeeze(0)
            latents = torch.from_numpy(latents_np).to(torch.float16)

            # Get prompts
            prompt = item['prompt']
            if args.prompt_prepend_content:
                prompt = args.prompt_prepend_content + prompt
            if args.prompt_append_content:
                prompt = prompt + args.prompt_append_content

            if "vae_audio" in self.task_kwargs["modality"] and "va" in task_type:
                audio, audio_success = self.get_audio_with_size(
                    idx,
                    return_type="vae",
                    column_name=video_path_col,
                )
                if not audio_success:
                    audio = None
            else:
                audio = None

            data = MultimodalDataContainer(
                prompt=prompt,
                videos=[self.vae_process_video(latents_np)],
                video_last_frames=None,
                audios=[audio] if audio is not None else None, 
                success=True,
                index=idx,
            )
            sections = self.build_t2v_template(data)

            dummy_type_dict = {}
            if self.args.audio_branch_model_name is not None and data.audios is None:
                dummy_type_dict["audio"] = 1    # audio branch dummy token
            if data.videos is None and data.images is None:
                dummy_type_dict["visual"] = 1   # visual branch dummy token
            dummy_number = sum(dummy_type_dict.values())

            max_token_length = self.seq_length - dummy_number \
                if self.sequence_pack \
                else self.max_token_length - dummy_number

            output = self.tokenizer.encode_general(
                sections=sections,
                max_token_length=max_token_length,
                add_eos=False,
                drop_last=self.drop_last,
                add_pad=False,
                add_bos=self.default_conv.add_bos if hasattr(self, 'default_conv') else True,
                und_token_type=self.und_token_type,
                gen_token_type=self.gen_token_type,
                disable_ignore=True,
            )
            output.text_mask = output.text_mask.to(torch.long)
            rope_media_info, _ = self.get_rope_media_info(sections, output, data)

            video_pair.append(latents)
            if audio is not None:
                audio_pair.append(audio)
            rope_pair.append(rope_media_info)

        loss_mode = "dpo"
        if "sft" in self.task_kwargs['dpo_mode']:
            win_source = item.get('win_source', 'leo')
            sft_sources = self.task_kwargs['sft_sources']
            for sft_source in sft_sources:
                if sft_source in win_source:
                    loss_mode = 'sft_dpo'
                    break

        result =  {
            "n_samples": 1,
            "loss_mode": loss_mode,
            "video_pair": video_pair,
            "rope_pair": rope_pair,
            "tokens": output.tokens.clone(),
            "text_mask": output.text_mask.clone(),
            "dataset_tag": self.dataset_tag,
            "task_type": task_type,
            "text": prompt,
            "dummy_type_dict": dummy_type_dict,
        }
        if len(audio_pair) == 2:
            result["audio_pair"] = audio_pair
        return result
        
    def __getitem__(self, idx):
        try_times = 10000
        for _ in range(try_times):
            try:
                return self.get_batch(idx)
            except Exception as e:
                self.logger.warning(f"Error loading sample {idx}: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError(f'Dataset errors occur {try_times} in continue __getitem__')


def dpo_loss(v_w_pred, v_l_pred, v_w_ref_pred, v_l_ref_pred, v_w_target, v_l_target, 
    beta=500, loss_mode=None, logger=None):
    
    reduce_dims = list(range(1, v_w_pred.ndim))

    model_w_err = (v_w_pred - v_w_target).pow(2).mean(dim=reduce_dims)
    model_l_err = (v_l_pred - v_l_target).pow(2).mean(dim=reduce_dims)
    ref_w_err = (v_w_ref_pred - v_w_target).pow(2).mean(dim=reduce_dims)
    ref_l_err = (v_l_ref_pred - v_l_target).pow(2).mean(dim=reduce_dims)

    # w_diff should decrease, l_diff should increase
    w_diff = model_w_err - ref_w_err
    l_diff = model_l_err - ref_l_err
    # w_diff - l_diff should be negative, decrease, inside_term should increase
    inside_term = -0.5 * beta * (w_diff - l_diff)

    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
    implicit_acc += 0.5 * (inside_term == 0).sum().float() / inside_term.size(0)

    loss = - torch.nn.functional.logsigmoid(inside_term)
    if loss_mode == "reg_dpo":
        dgr = 0.5 * beta * (1 - torch.sigmoid(inside_term))
        loss += dgr * w_diff
    elif loss_mode == "sft_dpo":
        loss += model_w_err
    
    extra_info = {
        "w_err": model_w_err.mean(),
        "l_err": model_l_err.mean(),
        "ref_w_err": ref_w_err.mean(),
        "ref_l_err": ref_l_err.mean(),
        "w_diff": w_diff.mean(),
        "l_diff": l_diff.mean(),
        "inside_term": inside_term.mean(),
        "implicit_acc": implicit_acc,
    }
    if logger is not None:
        logger.info(
            f"w_err: {extra_info['w_err']:.4f}, "
            f"l_err: {extra_info['l_err']:.4f}, "
            f"ref_w_err: {extra_info['ref_w_err']:.4f}, "
            f"ref_l_err: {extra_info['ref_l_err']:.4f}, "
            f"inside_term: {extra_info['inside_term']:.4f}, "
            f"implicit_acc: {extra_info['implicit_acc']:.4f}"
        )
    return loss.mean(), extra_info

def load_csv_when_done(proc, csv_path, dataset_name, monitor, global_step):
    while True:
        if proc.poll() is not None:  # Process has finished
            try:
                print(f"Process finished, loading results from {csv_path}...")
                df = pd.read_csv(csv_path)
                summary_events = []
                print(f"load evaliation csv df.columns: {df.columns.tolist()}")
                for col in df.columns:
                    if col.endswith('_mos'):
                        mos = df[col].mean()
                        if pd.notna(mos):
                            summary_events.append((f"Loss/{col}_{dataset_name}", mos, global_step))
                        else:
                            print(f"Skipping {col} for {dataset_name} because mean is NaN")
                print(f"Evaluation results for {dataset_name}: {summary_events}")
                monitor.write_events(summary_events)
            except Exception as e:
                print(f"Failed to load {csv_path}: {e}")
            break
        time.sleep(2)

def update_teacher_model(model, teacher_model, decay):
    #update teacher model, decay means how much to keep from the old teacher model, 0 means fully update to student model, 1 means keep the old teacher model
    with torch.no_grad():
        for src_param, tgt_param in zip(model.parameters(), teacher_model.parameters(), strict=True):
            src_data = src_param.detach().clone().data
            tgt_data = tgt_param.detach().data
            if type(tgt_data) != type(src_data):
                if type(tgt_data) is DTensor:
                    src_data = DTensor.from_local(src_data, tgt_data.device_mesh, tgt_data.placements)
                else:
                    src_data = src_data.full_tensor()
            tgt_param.data.copy_(tgt_param.detach().data * decay + src_data * (1.0 - decay))
    return teacher_model

class Leo2DPOTrainer(Leo2Trainer):
    def __init__(self, args):
        args.activation_offloading = True

        super().__init__(args)
        self.build_reference_model()
        self.tokenizer = load_tokenizer(args.tokenizer_name, args.tokenizer_class)
        def prompt_fn(prompt, _=None):
            # FPS:24,
            if args.prompt_prepend_content:
                prompt = args.prompt_prepend_content + prompt
            if args.prompt_append_content:
                prompt = prompt + args.prompt_append_content
            return {"role": "user", "content": prompt}
        self.prompt_fn = prompt_fn

        # freeze MOE gate layer
        count = 0
        for block in self.model_engine.layers:
            if isinstance(block.mlp, tuple(MOE_LAYER_IMPL.values())):
                for param in block.mlp.gate.parameters():
                    param.requires_grad = False
                    count += 1
        print(f'Freeze {count} moe gate parameters')

    def build_dataloader(self):
        # Build dataloaders for multimodal tasks
        self.task_info_dict = {
            't2vi2v': dict(cur_task="videodpo", cls=VideoDPOLatentDataset),
        }
        DatasetsProvider(self.task_info_dict)(None)
        self.combined_iterator = get_combined_iterator()
        self.mm_state = get_mm_state()

    def build_reference_model(self):
        args = self.args
        dtype = torch.bfloat16 if args.bf16 and not args.main_params_fp32 else torch.float32
        self.model_dtype = dtype
        self.teacher_model, self.teacher_model_config = build_model(
            args,
            dtype=dtype,
            device=args.init_device,
            initialize_weights=args.init_device != "meta",
        )
        self.teacher_model.collect_load_plans(
            None, args.teacher_checkpoint_dir,
            fuse_experts_in_load=args.fuse_experts_in_load,
            copy_mot_in_load=args.copy_mot_in_load and self.teacher_model_config.use_mot,
        )
        self.teacher_model.load_before_fsdp()

        # Build Model Engine
        ParallelEngine = find_engine(args.model_name)     # noqa
        self.teacher_model_engine = ParallelEngine(
            model=self.teacher_model,
            enable_autocast=args.autocast_dtype not in ["fp32", "float32"],
            autocast_prec=args.autocast_dtype,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            enable_gradient_checkpointing=args.recompute_granularity is not None,
            # Put meta param materialization into init_param_and_apply_fsdp2 for memory-efficient initialization
            initialize_meta_param=args.fsdp_impl == 'new',
            dp_replicate_param_handler='none',        
        )
        for plan in self.teacher_model.after_fsdp_plans:
            # Load from pretrained checkpoint
            if plan.source == "dcp":
                default_states = self.teacher_model_engine.pre_load_state_dict()
                self.teacher_model_engine.load_checkpoint(**plan.metadata)
                self.teacher_model_engine.post_load_state_dict(default_states)
            elif plan.source == "resume":
                self.teacher_model_engine.load_checkpoint(**plan.metadata)

        self.teacher_model_engine.eval()
        for param in self.teacher_model_engine.parameters():
            param.requires_grad = False
        
        if args.teacher_checkpoint_dir is None or not os.path.exists(args.teacher_checkpoint_dir):
            self.logger.info(f"Teacher model checkpoint dir {args.teacher_checkpoint_dir} not exists, set to student model weight")
            self.teacher_model_engine = update_teacher_model(self.model_engine, self.teacher_model_engine, decay=0.0)

    def prepare_model_inputs(self, batch, device):
        loss_mode = batch["loss_mode"][0]
        denoiser = get_denoiser()
        model_inputs = [] 
        v_t = None
        for i in range(2):
            micro_batch = copy.deepcopy(batch)
            micro_batch["videos"] = batch["video_pair"][i].to(device).unsqueeze(0)
            micro_batch["rope_media_info"] = [batch['rope_pair'][0][i]]
            if "audio_pair" in batch:
                micro_batch["audios"] = batch["audio_pair"][i].to(device).unsqueeze(0)
            tasktype2extend = {
                'i2va': 'i2v',
                't2va': 't2v',
            }
            extend_type = batch['task_type'][0]
            extend_type = tasktype2extend.get(extend_type, extend_type)
            with torch.autocast(device_type="cuda", enabled=False):
                model_input_kwargs, bsz, seqlen = prepare_model_t2v_inputs(micro_batch, device, 
                    latent_channel_extend_type=extend_type, timesteps=v_t)
            model_t = model_input_kwargs['timesteps']
            if v_t is None:
                v_t = denoiser.get_scheduler_t(model_t)
            model_inputs.append(model_input_kwargs)

        return loss_mode, model_inputs[0], model_inputs[1], bsz, seqlen

    @torch.no_grad()
    def sample_validation(self):
        args = self.args 
        run_task_kwargs = args.t2vi2v_task_kwargs
        index_task_kwargs = args.t2vi2v_index_kwargs

        from hymm.samplers.leo2_sampler import Leo2Sampler
        sampler = Leo2Sampler(None, self.rank, self.world_size, trainer=self)        
        #overwrite generation config with task specific kwargs if specified
        for key, value in run_task_kwargs.items():
            if hasattr(sampler.model.generation_config, key):
                setattr(sampler.model.generation_config, key, value)
        video_dirs = []
        save_dir = os.path.join(self.exp_dir, "samples", "step{}".format(self.ss.update_steps))
        os.makedirs(save_dir, exist_ok=True)
        for testset in index_task_kwargs['testsets']:
            dataset = MessageListDataset(
                testset,
                save_dir,
                tokenizer=self.tokenizer,
                prompt_fn=self.prompt_fn,
            )
            datasampler = DistributedSamplerFix(dataset, num_replicas=self.p_state.dp_size,
                                            rank=self.p_state.dp_rank, shuffle=False, drop_last=False,
                                            add_extra_samples="extend")
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=datasampler,
                                    drop_last=False, collate_fn=getattr(dataset, "collate_fn", None))
            save_base = dataset.save_dir

            timer = Timer(enabled=True)
            for batch_idx, batch in enumerate(dataloader):
                self.logger.info(f"Generating batch {batch_idx + 1} / {len(dataloader)} ...")
                timer.start(f"Batch")
                message_list=batch[dataset.name_mapper("message_list")]
                if "num_frames" in batch:
                    num_frames = int(batch["num_frames"][0])
                else:
                    num_frames = run_task_kwargs["num_frames"]
                outputs = sampler.model.generate_video(
                    message_list=message_list, seed=batch["seed"], 
                    video_size=run_task_kwargs["image_size"], 
                    num_frames=num_frames, 
                    video_fps=run_task_kwargs["video_fps"], 
                    ref_mode=run_task_kwargs["ref_mode"], 
                    output_type=dict(visual="np", audio="np"), 
                    bot_task=run_task_kwargs["bot_task"], 
                    verbose=1 if self.rank == 0 else 0
                )
                outputs = outputs.postprocess_outputs(batch)
                if self.p_state.cp_rank == 0:
                    outputs.save_to(
                        save_base=save_base,
                        summary_file_name=f"results/results_{self.p_state.dp_rank}.csv",
                        fps=run_task_kwargs['video_fps'],
                        sample_rate=args.audio_sample_rate,
                    )
                video_dir = os.path.join(save_base, "videos")
                # Log time
                timer.stop(f"Batch")
                self.logger.info(f"[Task {testset}] "
                        f"[{batch_idx + 1} / {len(dataloader)}] "
                        f"| {readable_time(timer, 'Batch', len(dataloader) - batch_idx - 1)} "
                        f"save to {save_dir}")
            video_dirs.append((testset, video_dir))

        return video_dirs

    def eval_validation(self, video_dirs):
        args = self.args
        task_args = args.t2vi2v_task_kwargs
        for (testset, video_dir) in video_dirs:
            dataset_name = os.path.basename(os.path.dirname(video_dir))
            out_csv_path = os.path.join(video_dir, "autoeval_result.csv")
            eval_cmd = task_args['eval_script'].format(video_dir=video_dir, out_csv_path=out_csv_path, label=testset)
            eval_cmds = eval_cmd.split(' ')
            print('running evaluation', eval_cmd)
            p=subprocess.Popen(eval_cmds)
            threading.Thread(
                target=load_csv_when_done, 
            args=(p, out_csv_path, dataset_name, self.model_engine.monitor, self.ss.update_steps)).start()
        return None

    @torch.no_grad()
    def validation(self, timer):
        video_dirs = self.sample_validation()
        # only the monitor writer rank runs the evaluation and loads the results
        if dist.get_rank() == self.model_engine.monitor.writer_rank:
            self.eval_validation(video_dirs)

    def train_step(self, batch):
        args = self.args
        task_args = args.t2vi2v_task_kwargs
        device = torch.device("cuda", args.local_rank)

        loss_mode, win_model_input_kwargs, lose_model_input_kwargs, bsz, seqlen = self.prepare_model_inputs(batch, device)
        win_ut = win_model_input_kwargs['ut']
        lose_ut = lose_model_input_kwargs['ut']
        train_audio = "audio_pair" in batch
        if train_audio:
            win_aut = win_model_input_kwargs['aut']
            lose_aut = lose_model_input_kwargs['aut']

        #calcuate loss
        with torch.no_grad():
            win_teacher_model_output = self.teacher_model_engine(**win_model_input_kwargs)
            lose_teacher_model_output = self.teacher_model_engine(**lose_model_input_kwargs)

        def dummy_loss_closure(model_output, model_input_kwargs):
            return model_output, {}
        self.model_engine.register_loss_closure(dummy_loss_closure)
        win_model_output = self.model_engine(**win_model_input_kwargs)

        def loss_closure(model_output, model_input_kwargs):
            lose_model_output = model_output
            loss, extra_info = dpo_loss(
                win_model_output.diffusion_prediction,
                lose_model_output.diffusion_prediction,
                win_teacher_model_output.diffusion_prediction,
                lose_teacher_model_output.diffusion_prediction,
                win_ut, lose_ut, 
                beta=int(task_args['dpo_beta']),
                loss_mode=loss_mode,
                logger=self.logger,
            )
            extra_info['vidoe_loss'] = loss
            if train_audio:
                aloss, _ = dpo_loss(
                    win_model_output.audio_diffusion_prediction,
                    lose_model_output.audio_diffusion_prediction,
                    win_teacher_model_output.audio_diffusion_prediction,
                    lose_teacher_model_output.audio_diffusion_prediction,
                    win_aut, lose_aut,
                    beta=int(task_args['dpo_beta']),
                    loss_mode=loss_mode,
                    logger=self.logger,
                )
                extra_info['audio_loss'] = aloss
                loss += aloss * args.audio_loss_weight
                extra_info['loss'] = loss
            return loss, extra_info

        self.model_engine.register_loss_closure(loss_closure)
        loss = self.model_engine(**lose_model_input_kwargs)
        model_output = self.model_engine.get_cached_result("ret_val")
        if torch.isnan(loss).any():
            self.nan_grad_count += 1
            self.logger.warning(f"NaN loss encountered in rank {self.rank}, total NaN count: {self.nan_grad_count}")

        loss_dict = self.model_engine.get_cached_result("loss_dict")
        loss_dict['loss'] = loss
        for key in loss_dict.keys():
            if key not in self.loss_names:
                self.loss_names.append(key)

        consumed_metrics = {
            batch["dataset_tag"][0]: {
                "samples": batch["n_samples"].sum().item(),
                "tokens": bsz * seqlen
            }
        }

        def backward_fn(loss):
            # HACK: 不能直接 loss.backward, 因为当前版本 pytorch 的特性会让 leaf node 推迟 reduce。
            grad_outputs = [
                win_model_output.diffusion_prediction,
                model_output.diffusion_prediction,
            ]
            if train_audio:
                grad_outputs.extend([
                    win_model_output.audio_diffusion_prediction,
                    model_output.audio_diffusion_prediction,
                ])
            grads = torch.autograd.grad(loss, grad_outputs, retain_graph=True)

            self.model_engine.set_is_last_backward(False)
            if train_audio:
                torch.autograd.backward(
                    [win_model_output.diffusion_prediction, win_model_output.audio_diffusion_prediction],
                    [grads[0], grads[2]],
                    retain_graph=False,
                )
            else:
                torch.autograd.backward(win_model_output.diffusion_prediction, grads[0], retain_graph=False)

            self.model_engine.set_is_last_backward(True)
            if train_audio:
                torch.autograd.backward(
                    [model_output.diffusion_prediction, model_output.audio_diffusion_prediction],
                    [grads[1], grads[3]],
                    retain_graph=False,
                )
            else:
                torch.autograd.backward(model_output.diffusion_prediction, grads[1], retain_graph=False)

        loss_dict["backward_fn"] = backward_fn

        return loss_dict, consumed_metrics