import gc
from functools import partial

import torch
import torch.distributed as dist
import torch.utils
from index_kits.sampler import BlockDistributedSampler
from torch.utils.data import DataLoader

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from hymm.config import *
from hymm.data_kits.samplers import SequentialSampler
from hymm.data_kits.text_image_iterator import TextImageBatchIterator
from hymm.data_kits.text_image_transfusion_loader import TransfusionTextImageArrowStream
from hymm.data_kits.text_loader import TextArrowStream, MaxLengthBatchSampler
from hymm.diffusion import load_denoiser
from hymm.models import DistributedEMA, TPDistributedEMA
from hymm.models import build_model
from hymm.models import load_vae
from hymm.ptm.training import pretrain
from hymm.utils.lr_schedules import add_tuning_arguments
from hymm.utils.torch_utils import PRECISION_TO_TYPE, set_worker_seed_builder, set_manual_seed
from megatron import get_args, get_timers, print_rank_0, mpu
from megatron.utils import average_losses_across_data_parallel_group

gc.set_threshold(7000, 100, 100)


def model_provider(pre_process=True, post_process=True):
    """ Build the model. """

    args = get_args()
    args = sanity_check_args(args)

    see_memory_usage(f"Before Building Model", force=True)
    factor_kwargs = {'device': torch.device("cuda", args.local_rank), 'dtype': PRECISION_TO_TYPE[args.precision]}

    assert args.deepspeed
    assert args.no_pipeline_parallel
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=False,
                             mpu=mpu):
        model, model_settings = build_model(args, **factor_kwargs)
    see_memory_usage(f"After Building Model", force=True)

    # After model initialization, we set different seed for each process.
    set_manual_seed(args.seed + mpu.get_data_parallel_rank())

    return model


def ema_provider(args, model):
    """
        Build DistributedEMA from model
    """
    if args.ptm_v2:
        ema = TPDistributedEMA(global_world_size=dist.get_world_size(),
                               global_rank=dist.get_rank(),
                               dp_world_size=mpu.get_data_parallel_world_size(),
                               dp_rank=mpu.get_data_parallel_rank(),
                               module=model[0].module,
                               dtype=args.ema_precision,
                               decay=args.ema_decay,
                               warmup=args.ema_warmup,
                               power=args.ema_warmup_power)
    else:
        ema = DistributedEMA(world_size=dist.get_world_size(),
                             rank=dist.get_rank(),
                             module=model[0].module,
                             dtype=args.ema_precision,
                             decay=args.ema_decay,
                             warmup=args.ema_warmup,
                             power=args.ema_warmup_power)
    # load ema weight from ckpt, don't support change DP ranks
    if not args.no_load_ema:
        load_dir = args.load
        tag = None
        if load_dir is None or not os.path.exists(load_dir):
            print_rank_0(f"load ema ckpt error: load_dir:{load_dir} not exist")
        else:
            dp_rank = mpu.get_data_parallel_rank()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            # get ema ckpt path
            while load_dir[-1] == '/':
                load_dir = load_dir[:-1]
            basename = os.path.basename(load_dir)
            if 'global_step' in basename:
                tag = basename
                load_dir = os.path.dirname(load_dir)
            else:
                latest_path = os.path.join(load_dir, "latest")
                if os.path.isfile(latest_path):
                    with open(latest_path, "r") as fd:
                        tag = fd.read().strip()
                else:
                    print(
                        f"in ema_provider, rank: {dist.get_rank()}, ERROR: Unable to find latest file at {latest_path}",
                        flush=True)
            print(f"in ema_provider, rank: {dist.get_rank()}, we use load_dir: {load_dir}, tag: {tag}", flush=True)

            ema_ckpt_name = f"{load_dir}/{tag}/dist_ema/ema_dp_rank_{dp_rank:02}_tp_rank_{tp_rank:02}_pp_rank_{pp_rank:02}.pt"
            if os.path.exists(ema_ckpt_name):
                device = torch.device("cuda", args.local_rank)
                ema_ckpt = torch.load(ema_ckpt_name, map_location=device)
                ema.load_ckpt(ema_ckpt)
                print(f"rank: {dist.get_rank()}, load ema from {ema_ckpt_name}", flush=True)
                # not sure if need barrier
                torch.distributed.barrier()
            else:
                print(f"rank: {dist.get_rank()}, load ema ckpt error: ema_ckpt_name: {ema_ckpt_name} not exist",
                      flush=True)
    return ema


# [vae, text_encoder, text_encoder_2, denoiser, ema]
def extra_models_provider(model):
    """ Build denoise scheduler, vae, text_encoder """
    args = get_args()
    device = torch.device("cuda", args.local_rank)

    print_rank_0("Building VAE...")
    vae = load_vae(
        args.vae_type,
        args.vae_precision,
        device=device,
    )

    # ====================== Build denoise scheduler ========================
    print_rank_0("Building denoise scheduler...")
    denoiser = load_denoiser(args)

    ema = None
    if args.use_ema:
        ema = ema_provider(args, model)
    text_encoder = None
    text_encoder_2 = None
    return vae, text_encoder, text_encoder_2, denoiser, ema


def broadcast_data(data_iterator):
    """PTM 中每个 DP group 只有 TP0 读取数据, 然后将 batch 数据 broadcast 到其他 TP rank.
    broadcast_data 需要处理视频数据和图像数据, 不同类型数据的处理逻辑不一样.

    NOTE:
        视频数据当 media 或者 latents 为 None 时, dtype 为 torch.int64; 不为 None 时, dtype 分别为:
            media: torch.float32
            latents: torch.float16

        图像数据 media 为 torch.float32, latents 为 torch.int64


    Video:
        media: torch.float32 / torch.int64
        latents: torch.float16 / torch.int64
        text_ids: torch.int64
        text_mask: torch.int64
        text_ids_2: torch.int64
        text_mask_2: torch.int64
        type: torch.int64


    Image:
        media: torch.float32
        latents: torch.int64
        text_ids: torch.int64
        text_mask: torch.int64
        text_ids_2: torch.int64
        text_mask_2: torch.int64
        type: torch.int64
    """

    class _DataType:
        Video = 0
        Image = 1

    class _Dtype:
        Float16 = 0
        Float32 = 1

    data_flag = {}

    if data_iterator is not None:
        ## ------------ get batch ------------
        batch = next(data_iterator)

        ## ------------ parse batch ------------
        media, latents, *batch_args = batch
        if len(batch_args) == 3:
            text_ids, text_mask, kwargs = batch_args
            text_ids_2, text_mask_2 = torch.tensor(0), torch.tensor(0)
            multitask_cond_flag = torch.tensor(0, dtype=torch.int64)
            use_multitask_cond_flag = torch.tensor(0, dtype=torch.int64)
        elif len(batch_args) == 5:
            text_ids, text_mask, text_ids_2, text_mask_2, kwargs = batch_args
            multitask_cond_flag = torch.tensor(0, dtype=torch.int64)
            use_multitask_cond_flag = torch.tensor(0, dtype=torch.int64)
        elif len(batch_args) == 4:
            text_ids, text_mask, multitask_cond_flag, kwargs = batch_args
            text_ids_2, text_mask_2 = torch.tensor(0), torch.tensor(0)
            use_multitask_cond_flag = torch.tensor(1, dtype=torch.int64)
        elif len(batch_args) == 6:
            text_ids, text_mask, text_ids_2, text_mask_2, multitask_cond_flag, kwargs = batch_args
            use_multitask_cond_flag = torch.tensor(1, dtype=torch.int64)
        else:
            raise ValueError(f"Unexpected batch_args.")

        data_flag = {"use_multitask_cond_flag": use_multitask_cond_flag,
                     "multitask_cond_flag": multitask_cond_flag}
        data_flag = mpu.broadcast_data(["use_multitask_cond_flag", "multitask_cond_flag"], data_flag, torch.int64)
        use_multitask_cond_flag = data_flag["use_multitask_cond_flag"]
        multitask_cond_flag = data_flag["multitask_cond_flag"]

        ## ------------ pack batch ------------
        # torch.int64
        data_a = {}

        # torch.int64
        data_b = {}
        data_b['text_ids'] = text_ids
        data_b['text_mask'] = text_mask
        data_b['text_ids_2'] = text_ids_2
        data_b['text_mask_2'] = text_mask_2

        if kwargs['type'][0] == "image":
            """ image 用 media 数据, latents 为 torch.tensor(0)
            """
            data_a['data_type'] = torch.tensor(_DataType.Image)
            data_a['data_dtype'] = torch.tensor(_Dtype.Float32)  # 记录 data_c 数据类型

            data_b['latents'] = latents

            # torch.float32
            data_c = {}
            data_c['media'] = media

        elif kwargs['type'][0] == "video":
            """ video 同时支持 media 和 latents 数据, 针对不同数据进行处理
            """
            data_a['data_type'] = torch.tensor(_DataType.Video)

            # torch.float32 (media) or torch.float16 (latents)
            data_c = {}

            if len(media.shape) == 1:
                """ 用 vae latents 数据, media 为 torch.tensor(0)
                """
                assert latents.dtype == torch.float16, "latents dtype should be torch.float16."
                data_a['data_dtype'] = torch.tensor(_Dtype.Float16)  # 记录 data_c 数据类型
                data_b['media'] = media  # torch.int64
                data_c['latents'] = latents  # torch.float16
            elif len(media.shape) == 5:
                """ 用原始视频数据, latents 为 torch.tensor(0)
                """
                assert media.dtype == torch.float32, "media dtype should be torch.float32."
                data_a['data_dtype'] = torch.tensor(_Dtype.Float32)  # 记录 data_c 数据类型
                data_c['media'] = media  # torch.float32
                data_b['latents'] = latents  # torch.int64
            else:
                raise ValueError(f"Unknown video media shape: {media.shape}")

        else:
            raise ValueError(f"Unknown batch type: {kwargs['type']}")

    else:
        batch = []
        data_a = None  # data_type, data_dtype
        data_b = None  # text_ids, text_mask, text_ids_2, text_mask_2, (media, latents)
        data_c = None  # media or latents
        data_flag = mpu.broadcast_data(["use_multitask_cond_flag", "multitask_cond_flag"], data_flag, torch.int64)
        use_multitask_cond_flag = data_flag["use_multitask_cond_flag"]
        multitask_cond_flag = data_flag["multitask_cond_flag"]

    use_multitask_cond_flag = bool(use_multitask_cond_flag.item())

    # Items and their type.
    keys = ['data_type', 'data_dtype']
    data_a = mpu.broadcast_data(keys, data_a, torch.int64)

    if data_a['data_type'] == _DataType.Image:

        keys = ['latents', 'text_ids', 'text_mask', 'text_ids_2', 'text_mask_2']
        datatype = torch.int64
        data_b = mpu.broadcast_data(keys, data_b, datatype)

        assert data_a['data_dtype'] == _Dtype.Float32

        keys = ['media']
        datatype = torch.float32
        data_c = mpu.broadcast_data(keys, data_c, datatype)

        # Unpack
        if mpu.get_tensor_model_parallel_rank() != 0:
            batch.append(data_c['media'])
            batch.append(data_b['latents'])

    elif data_a['data_type'] == _DataType.Video:

        if data_a['data_dtype'] == _Dtype.Float16:
            """ data_c 内保存的 latents """

            keys = ['media', 'text_ids', 'text_mask', 'text_ids_2', 'text_mask_2']
            datatype = torch.int64
            data_b = mpu.broadcast_data(keys, data_b, datatype)

            keys = ['latents']
            datatype = torch.float16
            data_c = mpu.broadcast_data(keys, data_c, datatype)

            # Unpack
            if mpu.get_tensor_model_parallel_rank() != 0:
                batch.append(data_b['media'])
                batch.append(data_c['latents'])

        elif data_a['data_dtype'] == _Dtype.Float32:
            """ data_c 内保存的 media """

            keys = ['latents', 'text_ids', 'text_mask', 'text_ids_2', 'text_mask_2']
            datatype = torch.int64
            data_b = mpu.broadcast_data(keys, data_b, datatype)

            keys = ['media']
            datatype = torch.float32
            data_c = mpu.broadcast_data(keys, data_c, datatype)

            # Unpack
            if mpu.get_tensor_model_parallel_rank() != 0:
                batch.append(data_c['media'])
                batch.append(data_b['latents'])

        else:
            raise ValueError(f"Unknown data_dtype: {data_a['data_dtype']}")

    else:
        raise ValueError(f"Unknown data_type: {data_a['data_type']}")

    if mpu.get_tensor_model_parallel_rank() != 0:
        batch.append(data_b['text_ids'])
        batch.append(data_b['text_mask'])
        batch.append(data_b['text_ids_2'] if len(data_b['text_ids_2'].shape) != 0 else None)
        batch.append(data_b['text_mask_2'] if len(data_b['text_mask_2'].shape) != 0 else None)
        if use_multitask_cond_flag:
            batch.append(multitask_cond_flag)
        kwargs = {}
        kwargs['type'] = ['image'] if data_a['data_type'] == _DataType.Image else ['video']
        batch.append(kwargs)

    return batch


def get_batch(batch, extra_models):
    if "image" in batch:
        inputs = prepare_model_image_inputs(batch, extra_models)
    elif "text" in batch:
        inputs = prepare_model_text_inputs(batch, extra_models)
    else:
        raise ValueError("Unknown batch type, expected 'image' or 'text'.")
    return inputs


def prepare_model_text_inputs(batch, extra_models):
    args = get_args()
    device = torch.device("cuda", args.local_rank)
    # [vae, text_encoder, text_encoder_2, denoiser, ema]
    vae, text_encoder, text_encoder_2, denoiser, _ = extra_models

    tokens = batch["tokens"][:, :-1].contiguous().to(device)
    target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
    text_mask = batch["text_mask"][:, 1:].contiguous().to(device)

    batch_size, n_tokens = tokens.shape

    # ==== Add dummy tokens to avoid hanging when deepspeed all-reduce gradients ====
    # Different modalities connect with different model parameters in the computation graph.
    # If different batches correspond to different modalities, deepspeed cannot correctly perform
    # all-reduce gradients. Therefore, we need to pad dummy image tokens to text sequences to
    # maintain consistent activated model parameters.
    dummy_tokens = torch.zeros((batch_size, args.dummy_number), dtype=tokens.dtype, device=device)
    dummy_target_tokens = (-100) * torch.ones((batch_size, args.dummy_number), dtype=tokens.dtype, device=device)
    tokens = torch.cat([tokens, dummy_tokens], dim=1)
    target_tokens = torch.cat([target_tokens, dummy_target_tokens], dim=1)

    image_mask = torch.zeros_like(tokens, dtype=torch.float32, device=device)
    image_mask[:, -1] = 1.0
    # Add iw,ih,timestep tokens to touch their embedding layers (include learnable parameters).
    if args.add_iw_ih_token:
        iw_ih_scatter_index = torch.tensor([[n_tokens, n_tokens + 1]] * batch_size, dtype=torch.long, device=device)
        iw_ih_scatter_src = torch.tensor([[2, 2]] * batch_size, dtype=torch.long, device=device)
    if args.add_timestep_token:
        timestep_scatter_index = torch.tensor([[n_tokens + 2]] * batch_size, dtype=torch.long, device=device)
    n_tokens += args.dummy_number

    text_mask = torch.cat([text_mask, torch.zeros_like(dummy_tokens, dtype=torch.float32, device=device)], dim=1)

    # Mixed attention mask
    _, n_tokens = tokens.shape
    causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
    attention_mask = causal_mask.view(1, 1, n_tokens, n_tokens).repeat(batch_size, 1, 1, 1)

    # Process dummy image tokens
    patch_size = args.patch_size
    latents = torch.randn((batch_size, args.vae_latent_dim, patch_size, patch_size), device=device)
    t, x_0, x_1 = denoiser.sample(latents, n_tokens)
    t, x_t, u_t = denoiser.path_sampler.plan(t, x_0, x_1)
    model_t = denoiser.get_model_t(t)  # t*1000

    model_intput_kwargs = dict(
        idx=tokens,  # [b, 512]
        target=target_tokens,  # [b, 512]
        attention_mask=attention_mask,  # [b, 512, 512]
        x_t=x_t,
        t=model_t,
        diffusion_loss_fn=partial(denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
        text_mask=text_mask,
        image_mask=image_mask,
        image_loss_weight=0,  # Set to zero to avoid dummy image tokens to affect the text loss
        data_type="text",
    )
    if args.add_iw_ih_token:
        model_intput_kwargs.update({
            "iw_ih_scatter_index": iw_ih_scatter_index,  # [b, 2]
            "iw_ih_scatter_src": iw_ih_scatter_src,  # [b, 2]
        })
    if args.add_timestep_token:
        model_intput_kwargs.update({
            "timestep_scatter_index": timestep_scatter_index,  # [b, 1]
        })
    return model_intput_kwargs, batch_size, n_tokens


def prepare_model_image_inputs(batch, extra_models):
    args = get_args()
    device = torch.device("cuda", args.local_rank)
    vae, text_encoder, text_encoder_2, denoiser, _ = extra_models

    # text: 256 + 1, image: 256, total: 513, 513 - 1 = 512
    tokens = batch["tokens"][:, :-1].contiguous().to(device)

    # ===================================== IMPORTANT =====================================
    # target_token is only used to calculate losses on text tokens and some special tokens
    # <img> is set to -100 in target_token
    target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
    text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
    # image_loss is computed inplace, therefore image_mask is shifted same as tokens
    image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

    # build attention mask
    batch_size = tokens.shape[0]
    n_tokens = tokens.shape[1]
    causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
    causal_mask = causal_mask.view(1, n_tokens, n_tokens).repeat(batch_size, 1, 1)
    image_mask_1 = image_mask.view(batch_size, 1, n_tokens).repeat(1, n_tokens, 1)
    image_mask_2 = image_mask_1.transpose(1, 2)
    attention_mask = causal_mask | (image_mask_1.bool() & image_mask_2.bool())
    # unsqueeze for attention head dim
    attention_mask = attention_mask.unsqueeze(1)

    # ===================================== prepare diffusion =====================================
    image = batch["image"].to(device)
    vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
    with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
        vae_encode_result = vae.encode(image)
        if isinstance(vae_encode_result, torch.Tensor):
            latents = vae_encode_result
        else:
            latents = vae_encode_result.latent_dist.sample()
        if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor:
            latents.sub_(vae.config.shift_factor)  # .mul_(self.vae.config.scaling_factor)
        if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor:
            latents.mul_(vae.config.scaling_factor)

    # b c t h w
    if hasattr(vae, "ffactor_temporal"):
        assert latents.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
        latents = latents.squeeze(2)

    t, x_0, x_1 = denoiser.sample(latents, n_tokens)
    t, x_t, u_t = denoiser.path_sampler.plan(t, x_0, x_1)
    diffusion_loss_fn = partial(denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t)
    model_t = denoiser.get_model_t(t)  # t*1000

    # ===================================== Pack model kwargs =====================================
    model_intput_kwargs = dict(
        idx=tokens,  # [b, 512]
        x_t=x_t,  # [b, c, h, w]
        t=model_t,  # [b]
        diffusion_loss_fn=diffusion_loss_fn,
        target=target_tokens,  # [b, 512]
        text_mask=text_mask,  # [b, 512]
        image_mask=image_mask,  # [b, 512]
        attention_mask=attention_mask,  # [b, 512, 512]
        image_loss_weight=args.image_loss_weight,
        data_type="image",
    )
    if args.add_iw_ih_token:
        assert "iw_ih_scatter_index" in batch and "iw_ih_scatter_src" in batch, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"
        model_intput_kwargs.update({
            "iw_ih_scatter_index": batch["iw_ih_scatter_index"].to(device),  # [b, 2]
            "iw_ih_scatter_src": batch["iw_ih_scatter_src"].to(device),  # [b, 2]
        })
    if args.add_timestep_token:
        assert "timestep_scatter_index" in batch, "timestep_scatter_index is required for adding timestep token"
        model_intput_kwargs.update({
            "timestep_scatter_index": batch["timestep_scatter_index"].to(device),  # [b, 1]
        })
    if args.rope_type in ['3d', '3d-interleave']:
        freqs_cos = batch["freqs_cos"].to(device)
        freqs_sin = batch["freqs_sin"].to(device)
        model_intput_kwargs.update(dict(
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        ))

    return model_intput_kwargs, batch_size, n_tokens


def loss_func(loss_dict, output_tensor):
    loss = loss_dict['loss'].mean().float()
    # text_loss
    if 'text_loss' in loss_dict:
        text_loss = loss_dict['text_loss'].mean().float()
        text_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    else:
        text_loss = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        text_count = torch.tensor(0, dtype=loss.dtype, device=loss.device)
    # image_loss
    if 'image_loss' in loss_dict:
        image_loss = loss_dict['image_loss'].mean().float()
        image_count = torch.tensor(1, dtype=loss.dtype, device=loss.device)
    else:
        image_loss = torch.zeros(loss.shape, dtype=loss.dtype, device=loss.device)
        image_count = torch.tensor(0, dtype=loss.dtype, device=loss.device)

    averaged_losses = average_losses_across_data_parallel_group([loss, text_loss, image_loss, text_count, image_count])
    return loss, {'lm loss': averaged_losses[0], 'text loss': averaged_losses[1] / averaged_losses[3], 'image loss': averaged_losses[2] / averaged_losses[4]}


def forward_step(data_iterator, model, extra_models, valid=False):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    ### Broadcast data.
    # batch = broadcast_data(data_iterator)
    batch = next(data_iterator)
    model_input_kwargs, cur_batch_size, n_tokens = get_batch(batch, extra_models)

    # torch.distributed.barrier()
    timers('batch-generator').stop()

    # valid = args.eval_interval and args.iteration % args.eval_interval == 0
    # if args.offline_sample_video or args.only_validation or valid:
    #     log_validation(batch, model, extra_models)
    #     # validation before train, and exit after validation
    #     if args.only_validation or args.offline_sample_video:
    #         sys.exit()

    # [vae, text_encoder, text_encoder_2, denoiser, ema]
    _, _, _, denoiser, _ = extra_models

    # Predict the noise residual
    timers('model-forward').start()

    with torch.autocast(device_type="cuda", dtype=PRECISION_TO_TYPE[args.precision], enabled=True):
        loss_dict = model(**model_input_kwargs)

    timers('model-forward').stop()

    # loss_mean = loss_dict["loss"].detach().mean().float()
    # if args.iteration > 1000 and loss_mean > 0.6:
    #     print(f"Abnormal loss detected: {loss_mean}  "
    #           f"Rank: {mpu.get_data_parallel_rank()}  "
    #           f"Global step: {args.iteration}")

    output_tensor = None
    return output_tensor, partial(loss_func, loss_dict)


def train_valid_test_datasets_provider(train_val_test_num_samples, extra_models):
    """Build train, valid, and test datasets."""
    args = get_args()
    # args.image_dataset = None
    print_rank_0('> building train datasets for AR ...')

    image_dataset = TransfusionTextImageArrowStream(
        args=args,
        index_file=args.index_file,
        training_image_size=args.training_image_size,
        image_token_length=args.image_token_length,
        text_token_length=args.text_token_length,
        uncond_p=args.uncond_p,
        tokenizer_name=args.tokenizer_name,
        multireso=args.multireso,
        index_kwargs=dict(
            ceph_base=args.ceph_base,
            batch_size=1 if args.mix_scale else args.micro_batch_size,
            world_size=1 if args.mix_scale else mpu.get_data_parallel_world_size(),
            image_caption_rate=args.image_caption_rate,
            image_text_arrow_suffix=args.image_text_arrow_suffix,
            image_text_col=args.image_text_col,
            image_caption_arrow_suffix=args.image_caption_arrow_suffix,
            image_caption_col=args.image_caption_col,
            caption_sample_ratio=args.caption_sample_ratio,
            index_strategy=args.index_strategy,
        ),
        debug=False,
    )
    # args.image_dataset = image_dataset

    args.dummy_number = 1 + (2 if args.add_iw_ih_token else 0) + (
        1 if args.add_timestep_token else 0)
    t2t_max_length = args.text_token_length + args.image_token_length + 1 - args.dummy_number
    text_dataset = TextArrowStream(
        args=args,
        index_file=args.text_index_file,
        t2t_text_token_length=t2t_max_length,
        tokenizer_name=args.tokenizer_name,
        index_kwargs=dict(
            ceph_base=args.ceph_base,
        ),
    )

    print_rank_0("> finished creating AR datasets ...")

    train_dataset = [image_dataset, text_dataset]
    valid_dataset = None
    test_dataset = None

    return train_dataset, valid_dataset, test_dataset


def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset.
    """
    args = get_args()
    args.image_sampler = None
    args.video_sampler = None

    image_dataset, text_dataset = dataset[0], dataset[1]
    video_loader, image_loader = None, None

    ## 如果使用了 use_cache，那么 persistent_workers 必须设置为 False，否则 dataset 加载的 cached shuffle 无法被 worker 获取，导致出错。
    loader_kwargs = dict(num_workers=args.num_workers,
                         pin_memory=True,
                         prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
                         worker_init_fn=set_worker_seed_builder(args.local_rank),
                         persistent_workers=False)
    if image_dataset is not None:
        # Build sampler and data loader
        image_sampler = BlockDistributedSampler(
            image_dataset,
            num_replicas=mpu.get_data_parallel_world_size(),
            rank=mpu.get_data_parallel_rank(),
            shuffle=False,
            seed=args.seed,
            drop_last=True,
            align=args.micro_batch_size,
        )

        image_dataloader = DataLoader(image_dataset,
                                      batch_size=args.micro_batch_size,
                                      sampler=image_sampler,
                                      shuffle=False,
                                      drop_last=True,
                                      **loader_kwargs
                                      )

        # args.image_sampler = image_sampler

    if text_dataset is not None:
        t2t_max_length = args.text_token_length + args.image_token_length + 1
        text_sampler = SequentialSampler(text_dataset)
        text_batch_sampler = MaxLengthBatchSampler(
            text_dataset.index_manager, text_sampler, batch_size=args.micro_batch_size,
            max_length=t2t_max_length,
            length_getter=lambda idm, ind: idm.get_attribute(ind, 'hy_ids_length'),
        )
        text_dataloader = DataLoader(text_dataset, batch_sampler=text_batch_sampler, **loader_kwargs)

    data_loader = TextImageBatchIterator(
        ss=args,    # ss states are defined in args in PTM.
        fast_shuffle=args.fast_shuffle,
        rank=mpu.get_data_parallel_rank(),
        world_size=mpu.get_data_parallel_world_size(),
        text_dataset=text_dataset,
        text_sampler=text_sampler,
        text_dataloader=text_dataloader,
        image_dataset=image_dataset,
        image_sampler=image_sampler,
        image_dataloader=image_dataloader,
        text_sampling_prob=args.text_sampling_prob,
        initial_seed=args.seed,
    )

    return data_loader


def add_dit_args(parser: argparse.ArgumentParser):
    kwargs = dict(ptm=True)
    parser = add_logging_args(parser, **kwargs)
    parser = add_model_args(parser, **kwargs)
    parser = add_extra_models_args(parser, **kwargs)
    parser = add_denoise_schedule_args(parser, **kwargs)
    parser = add_data_args(parser, **kwargs)
    parser = add_deepspeed_args(parser, **kwargs)
    parser = add_ema_args(parser, **kwargs)
    parser = add_training_args(parser, **kwargs)
    parser = add_evaluation_args(parser, **kwargs)
    parser = add_tools_args(parser, **kwargs)
    parser = add_tuning_arguments(parser, **kwargs)

    group = parser.add_argument_group(title="dit")
    group.add_argument("--build_pretraining_data_loader_func", default=build_pretraining_data_loader)
    # 开启 log_validation, 不开启的话不会在训练中validation, ${EVAL_INTERVAL}也不会生效
    group.add_argument("--dit-valid", action="store_true", default=False, help="whether validation in training")
    # 关闭 autocast
    group.add_argument("--disable-torch-amp", action="store_true", help="disable torch amp.")
    # 使用 fp32 算子
    group.add_argument("--force-fp32-ops", action='store_true')
    # 使用 TE 算子, 开启 TP / SP
    group.add_argument("--ptm-v2", action="store_true", help="enable ptm v2.")
    # 开启 FusedAttention 优化
    group.add_argument('--use-fused-attn', action='store_true', help='use FusedAttention implementation of attention.')
    # sequence-parallel padding
    group.add_argument("--sequence-parallel-padding", action="store_true", help="padding img for sequence parallel")
    group.add_argument("--sp-img-pad-size", type=int, default=-1,
                       help="img padded size for sequence parallel, don't need setting.")
    group.add_argument("--sp-txt-pad-size", type=int, default=-1,
                       help="txt padded size for sequence parallel, don't need setting.")
    group.add_argument("--sp-x-pad-size", type=int, default=-1,
                       help="x padded size for sequence parallel, don't need setting.")
    # 评估流程使用，只跑validation不训练, 优先级最高的参数, 谨慎开启
    group.add_argument("--only-validation", action="store_true", default=False,
                       help="validation before train, and exit after validation.")
    # 评估流程使用，只跑视频的sample validation不训练
    group.add_argument("--offline-sample-video", action="store_true", default=False,
                       help="validation before train, and exit after validation.")
    group.add_argument('--consumed-video', type=int, default=0, help='consumed video')
    group.add_argument('--consumed-img1', type=int, default=0, help='consumed img1')
    group.add_argument('--consumed-img2', type=int, default=0, help='consumed img2')
    group.add_argument('--consumed-img3', type=int, default=0, help='consumed img3')
    group.add_argument('--dummy-number', type=int, default=0, help='dummy number')
    group.add_argument("--vae-encode-chunk-size", type=int, default=-1, help="vae encode chunk size.")
    # choose some ops to not be recomputed
    group.add_argument('--selective-checkpoint', action='store_true', help='choose some ops to not be recomputed')
    ## a list of op names which not be recomputed
    # parser.add_argument('--skip-recompute-ops', nargs='+', type=str, default=['op_fused_attn_fwd.default'], help='a list of op names which not be recomputed')
    # layers enable selective-checkpoint
    group.add_argument('--selective-checkpoint-layers-range', type=str,
                       help='layers enable selective-checkpoint, for example: 3-5 means [3, 5)')
    # force enable qkv_weight_interleaved when ptm_v2 and tp_size = 1
    group.add_argument('--force-enable-qkv-interleaved', action='store_true',
                       help='force enable qkv_weight_interleaved when ptm_v2 and tp_size = 1')

    return parser


if __name__ == "__main__":
    pretrain(train_valid_test_dataset_provider=train_valid_test_datasets_provider,
             model_provider=model_provider,
             forward_step_func=forward_step,
             extra_args_provider=add_dit_args,
             extra_models_provider=extra_models_provider,
             )
