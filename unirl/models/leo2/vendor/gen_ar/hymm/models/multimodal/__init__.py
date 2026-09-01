from loguru import logger


def build_model(args, dtype=None, device=None, **kwargs):
    factory_kwargs = {"device": device or 'cpu', "dtype": dtype}

    model_structure = args.model_structure
    valid_keywords = {"initialize_weights"}
    valid_kwargs = {key: value for key, value in kwargs.items() if key in valid_keywords}

    if model_structure == "HunyuanMultimodal":
        from .hunyuan_multimodal import HunyuanMultimodal
        from .hunyuan_multimodal_config import HunyuanMultimodalConfig, core_model_config_from_args

        model_name = args.model_name.split(".")[-1]
        model_config_dict = core_model_config_from_args(args)
        model_config = HunyuanMultimodalConfig.from_name(model_name, **model_config_dict)
        # for heterogeneous branches for mot
        if args.gen_branch_model_name is not None and args.gen_branch_model_name != model_name:
            gen_config = HunyuanMultimodalConfig.from_name(args.gen_branch_model_name)
        else:
            gen_config = None
        model = HunyuanMultimodal(args, model_config, **factory_kwargs, **valid_kwargs, gen_config=gen_config)

    elif model_structure == "HunyuanMultimodalHF":
        from .hunyuan_multimodal_hf import HunyuanMultimodalHF
        from .hunyuan_multimodal_config import HunyuanMultimodalConfig, core_model_config_from_args

        model_name = args.model_name.split(".")[-1]
        model_config_dict = core_model_config_from_args(args)
        model_config = HunyuanMultimodalConfig.from_name(model_name, **model_config_dict)
        # for heterogeneous branches for mot
        if args.gen_branch_model_name is not None and args.gen_branch_model_name != model_name:
            gen_config = HunyuanMultimodalConfig.from_name(args.gen_branch_model_name)
        else:
            gen_config = None
        model = HunyuanMultimodalHF(args, model_config, **factory_kwargs, **valid_kwargs, gen_config=gen_config)

    elif model_structure == "HunyuanMultimodalMCore":
        from megatron.training.arguments import core_transformer_config_from_args
        from .hunyuan_multimodal_mcore import HunyuanMultimodalMCoreConfig, HunyuanMultimodalMCore
        from .hunyuan_multimodal_config import core_model_config_from_args

        model_name = args.model_name.split(".")[-1]
        model_config_dict = {
            **core_transformer_config_from_args(args),
            **core_model_config_from_args(args),
        }
        model_config = HunyuanMultimodalMCoreConfig.from_name(model_name, **model_config_dict)
        model = HunyuanMultimodalMCore(args, model_config, **factory_kwargs, **valid_kwargs)

    else:
        raise NotImplementedError(f"Model structure {model_structure} not implemented.")

    logger.info(f"Build Model {model.__class__.__name__} finished.")

    return model, model_config
