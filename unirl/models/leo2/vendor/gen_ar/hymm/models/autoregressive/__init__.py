# Implement building of autoregressive models and loading huggingface checkpoints.
#


def load_hf_model(args, model_config, ckpt_path, logger, device=None, dtype=None):
    import json
    import re
    import torch
    from accelerate.big_modeling import dispatch_model
    from transformers.modeling_utils import _get_resolved_checkpoint_files
    from transformers.utils.generic import ContextManagers
    from transformers import AutoConfig

    from hymm.utils.import_utils import is_package_version
    from hymm.utils.env import should_use_external_model
    from .hunyuan import HunYuanPreTrainedModel
    from .multimodal_transfusion import MultiModalTransfusionHF

    # Read config from huggingface ckpt directory
    hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    # 临时修复一些 ptm 导出时没对齐的配置
    hf_config.eod_token_id = 127957
    hf_config.eos_token_id = 127957
    hf_config.norm_topk_prob = True
    hf_config.pad_token_id = 128009
    hf_config.routed_scaling_factor = 1.0
    hf_config.moe_drop_tokens = not args.no_drop_tokens
    hf_config.use_fused_moe = args.infer_use_fused_moe
    hf_config.use_flash_attn = args.infer_use_flash_attn
    hf_config.capacity_factor = args.capacity_factor

    model_init_context = HunYuanPreTrainedModel.get_init_context(
        is_quantized=False, _is_ds_init_called=False,
    )

    # Load checkpoint files
    kwargs = {}
    if is_package_version("transformers", ">=", "4.53"):
        kwargs = dict(is_remote_code=False)
    checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=ckpt_path,
        subfolder='',
        variant=None,
        gguf_file=None,
        from_tf=False,
        from_flax=False,
        use_safetensors=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        local_files_only=False,
        token=False,
        user_agent={'file_type': 'model', 'framework': 'pytorch', 'from_auto_class': False},
        revision='main',
        commit_hash=None,
        **kwargs,
    )
    key_mapping = {
        r"model\.layers\.": "language_model.transformer.h.",
        r"layernorm_mlp\.mlp": "mlp",
    }

    # build model with meta device
    logger.info(f"Building MultiModalTransfusion model")
    kwargs = {}
    if args.pp_size > 1:
        kwargs['pp_size'] = args.pp_size
        kwargs['start_device_id'] = torch.distributed.get_rank() % torch.cuda.device_count()
    with ContextManagers(model_init_context):
        model = MultiModalTransfusionHF(hf_config, args, model_config, dtype, **kwargs)
    logger.info(f"Build finished.")

    # Make sure to tie the weights correctly
    # model.tie_weights()
    # Gemini 不共享权重
    logger.info(f"Calculating device map for model {model.__class__.__name__}")
    logger.info(f"Device map: \n{json.dumps(model.device_map, indent=4)}")

    # Load the state dict
    logger.info(f"Loading state dict from {ckpt_path}")

    use_ext_model_ = should_use_external_model()
    if use_ext_model_:
        # 清空language_model.transformer.h 的所有子层,避免加载.
        model.language_model.transformer.h = torch.nn.ModuleList()

    if is_package_version("transformers", ">=", "4.57"):
        kwargs = dict(
            keep_in_fp32_regex=re.compile(r"\.gate\.wg")
        )
    else:
        kwargs = dict(
            low_cpu_mem_usage=True,
            offload_state_dict=False,
            keep_in_fp32_modules=["mlp.gate.wg"] if args.gate_precision == "fp32" else [],
            _fast_init=True,
        )
    (
        model,
        missing_keys,
        unexpected_keys,
        mismatched_keys,
        offload_index,
        error_msgs,
    ) = HunYuanPreTrainedModel._load_pretrained_model(
        model=model,
        state_dict=None,
        checkpoint_files=checkpoint_files,
        pretrained_model_name_or_path=ckpt_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=sharded_metadata,
        device_map=model.device_map,
        disk_offload_folder=None,
        dtype=torch.bfloat16,
        hf_quantizer=None,
        device_mesh=None,
        key_mapping=key_mapping,
        weights_only=True,
        **kwargs,
    )

    # 使用外部模型推理时，会卸载language_model.transformer.h，避免打日志过多。
    if not use_ext_model_:
        logger.info(f"missing_keys: {missing_keys}")
        logger.info(f"unexpected_keys: {unexpected_keys}")
    logger.info(f"State dict loaded.")

    # dispatch model to devices
    device_map_kwargs = {
        "device_map": model.device_map,
        "offload_dir": None,
        "offload_index": offload_index,
        "offload_buffers": False,
        "skip_keys": model._skip_keys_device_placement,
    }
    dispatch_model(model, **device_map_kwargs)

    return model


def build_model(args, pretrained_ckpt=None, logger=None, dtype=None, device=None, **kwargs):
    """build model from config, handling loading weights in the model itself

    Returns
    -------
    model_: a torch model which forward must return a dict of outputs, containing "logits" in prediction mode
    and "loss" in training mode
    model_config: config that generates this model
    """
    from .config import Config
    from ...constants import PRETRAINED_LLM_PATH
    from ...utils.torch_utils import load_state_dict

    enable_meta = False # WIP: Not fully implemented, hard code to False for now
    if enable_meta and (args.pretrained_ckpt is not None or pretrained_ckpt is not None):
        factory_kwargs = {"device": 'meta', "dtype": dtype}
    else:
        factory_kwargs = {"device": 'cpu', "dtype": dtype}

    model_config = Config.from_name(args.model_name)
    model_kwargs = getattr(args, 'model_kwargs', {})
    if isinstance(model_kwargs, str):
        import json
        model_kwargs = json.loads(model_kwargs)
    model_config.update(model_kwargs)
    model_structure = getattr(args, "model_structure", "DiscreteMultiModalGPT")
    if model_structure == "DiscreteMultiModalGPT":
        from .model import DiscreteMultiModalGPT
        model_ = DiscreteMultiModalGPT(args, model_config, **factory_kwargs)
    elif model_structure == "ContinuousMultiModalGPT":
        from .model import ContinuousMultiModalGPT
        model_ = ContinuousMultiModalGPT(args, model_config, **factory_kwargs)
    elif model_structure == "ContinuousMultiHeadGPT":
        from .model import ContinuousMultiHeadGPT
        model_ = ContinuousMultiHeadGPT(args, model_config, **factory_kwargs)
    elif model_structure == "MarGPT":
        from .mar import MarGPT
        model_ = MarGPT(args, model_config, **factory_kwargs)
    elif model_structure == "MarTEGPT":
        from .mar_te import MarTEGPT
        model_ = MarTEGPT(args, model_config, **factory_kwargs)
    elif model_structure == "Transfusion":
        from .transfusion import Transfusion
        model_ = Transfusion(args, model_config, **factory_kwargs)
    elif model_structure == "MultiModalTransfusion":
        if getattr(args, "use_hf", None):
            if "ckpt_path" not in kwargs:
                raise ValueError("When use_hf is True, ckpt_path must be provided in kwargs")
            model_ = load_hf_model(args, model_config, ckpt_path=kwargs["ckpt_path"], logger=logger, **factory_kwargs)
        else:
            from .multimodal_transfusion import MultiModalTransfusion
            model_ = MultiModalTransfusion(args, model_config, **factory_kwargs)
    elif model_structure == "Dense":
        from .dense import GPT, gpt_config_from_args
        model_config = gpt_config_from_args(args)
        model_ = GPT(args, model_config, **factory_kwargs)
    else:
        raise NotImplementedError(f"model type {model_structure} not implemented")
    logger.info(f"Use model {model_.__class__.__name__}")

    if pretrained_ckpt is not None:
        args.pretrained_ckpt = pretrained_ckpt

    # When using pure_torch, the model ckpt is saved via 'dcp'.
    use_pure_torch = hasattr(args, "launcher") and args.launcher == "pure_torch"
    # When use_ptm==True, the model ckpt is loaded from "--load" in jobs_ptm/run_train_image_Gemini_*.sh
    use_ptm_load = getattr(args, 'use_ptm', False)
    # When use_hf==True, the model ckpt is loaded from huggingface ckpt.
    use_hf_load = getattr(args, 'use_hf', False)

    if use_pure_torch or use_ptm_load or use_hf_load:
        use_legacy_load = False
    else:
        use_legacy_load = True

    if use_legacy_load and getattr(args, "pretrained_ckpt", None) and not getattr(args, "no_load_pretrained", False):
        if args.pretrained_ckpt in PRETRAINED_LLM_PATH:
            pretrained_ckpt = PRETRAINED_LLM_PATH[args.pretrained_ckpt]
        else:
            pretrained_ckpt = args.pretrained_ckpt
        logger.info(f"loading pretrained checkpoint {pretrained_ckpt}")
        m, u = load_state_dict(
            model_, pretrained_ckpt,
            load_llm_to_mllm=args.get('load_llm_to_mllm', False),
            expand_keys=args.get('expand_keys'),
            load_prepend_key=args.get('load_prepend_key'),
            load_prepend_key_dict=args.get('load_prepend_key_dict'),
            shrink_keys=args.get('shrink_keys', None)
        )
        logger.info(
            f"loaded pretrained checkpoint,\n missing keys: {m}\nunexpected keys: {u}"
        )
    if use_pure_torch:
        from hymm.parallelism.parallel_states import get_parallel_state
        parallel_dims = get_parallel_state()

        # 如果有并行，模型一开始就是切好的，不能在这里读了
        if parallel_dims.ep or parallel_dims.pp:
            return model_, model_config

        logger.info(f"loading pretrained checkpoint {pretrained_ckpt}")
        from hymm.parallelism import checkpoint_manager
        from hymm.utils.torch_utils import expand_head_embedding, shrink_head_embedding

        def post_process_sd_fn(state_dict):
            load_llm_to_mllm=args.get('load_llm_to_mllm', False)
            expand_keys=args.get('expand_keys')
            load_prepend_key=args.get('load_prepend_key')
            load_prepend_key_dict=args.get('load_prepend_key_dict')
            shrink_keys=args.get('shrink_keys', None)
            if "module" in state_dict:
                state_dict = state_dict["module"]
            if load_llm_to_mllm:
                if expand_keys is None:
                    expand_keys = ["transformer.wte.weight", "lm_head.weight", "lm_head.bias"]
                state_dict = expand_head_embedding(model.state_dict(), state_dict, expand_keys)
            if load_prepend_key:
                state_dict = {load_prepend_key + k: v for k, v in state_dict.items()}
            if load_prepend_key_dict:
                new_state_dict = {}
                old_name_key_list = load_prepend_key_dict[0]
                new_prepend_key_list = load_prepend_key_dict[1]
                for k, v in state_dict.items():
                    new_key = k
                    for old_name_key, new_prepend_key in zip(old_name_key_list, new_prepend_key_list):
                        if old_name_key in k:
                            new_key = new_key.replace(old_name_key, new_prepend_key)
                    new_state_dict[new_key] = v
                state_dict = new_state_dict
            if shrink_keys is not None:
                state_dict = shrink_head_embedding(model.state_dict(), state_dict, shrink_keys)
            return state_dict
        checkpoint_manager.load_ckpt(model_, args.pretrained_ckpt, post_process_sd_fn=post_process_sd_fn)
    return model_, model_config
