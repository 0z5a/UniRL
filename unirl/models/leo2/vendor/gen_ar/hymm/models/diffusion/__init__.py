from easydict import EasyDict

import torch

from .dit import DiT, DiT_CONFIG
from .aries import Aries, AriesHF, Aries_CONFIG, MMDiT_CONFIG
from .vqunet import VQUNet, VQUNet_CONFIG
from .tv2a_aries import TV2A_Aries, TV2A_Aries_CONFIG


def build_model(args, pretrained_model=None, logger=None, dtype=None, device=None, **kwargs):
    factor_kwargs = {"device": device, "dtype": dtype}

    model_structure = getattr(args, "model_structure", None)

    if model_structure == "LeoModel":
        from .leo import LeoModel
        from .leo_config import LeoConfig, core_model_config_from_args

        model_name = args.model_name.split(".")[-1]
        model_config_dict = core_model_config_from_args(args)
        model_config = LeoConfig.from_name(model_name, **model_config_dict)

        if args.text_branch_model_name is not None:
            text_branch_config_dict = core_model_config_from_args(args, prefix="text_branch_")
            text_branch_model_name = args.text_branch_model_name.split(".")[-1]
            text_branch_config = LeoConfig.from_name(text_branch_model_name, **text_branch_config_dict)
        else:
            text_branch_config = None

        if args.audio_branch_model_name is not None:
            audio_branch_config_dict = core_model_config_from_args(args, prefix="audio_branch_")
            audio_branch_model_name = args.audio_branch_model_name.split(".")[-1]
            audio_branch_config = LeoConfig.from_name(audio_branch_model_name, **audio_branch_config_dict)
        else:
            audio_branch_config = None

        model = LeoModel(
            args, model_config, txt_config=text_branch_config, audio_config=audio_branch_config, 
            **factor_kwargs
        )
        return model, dict(main_branch=model_config, text_branch=text_branch_config, audio_branch=audio_branch_config)

    elif model_structure == "LeoModelHF":
        from .leo_hf import LeoModelHF
        from .leo_config import LeoConfig, core_model_config_from_args

        model_name = args.model_name.split(".")[-1]
        model_config_dict = core_model_config_from_args(args)
        model_config = LeoConfig.from_name(model_name, **model_config_dict)
        # for heterogeneous branches for mot
        if args.text_branch_model_name is not None:
            text_branch_config_dict = core_model_config_from_args(args, prefix="text_branch_")
            text_branch_model_name = args.text_branch_model_name.split(".")[-1]
            text_branch_config = LeoConfig.from_name(text_branch_model_name, **text_branch_config_dict)
        else:
            text_branch_config = None
        if args.audio_branch_model_name is not None:
            audio_branch_config_dict = core_model_config_from_args(args, prefix="audio_branch_")
            audio_branch_model_name = args.audio_branch_model_name.split(".")[-1]
            audio_branch_config = LeoConfig.from_name(audio_branch_model_name, **audio_branch_config_dict)
        else:
            audio_branch_config = None

        model = LeoModelHF(
            args, model_config, txt_config=text_branch_config, audio_config=audio_branch_config,
            **factor_kwargs
        )
        return model, dict(main_branch=model_config, text_branch=text_branch_config, audio_branch=audio_branch_config)

    # ==== Below is the legacy implementation for loading model. ====
    # In the future, we can consider to unify the model loading interface and remove the legacy implementation.
    model_name = args.model_name.split(".")[-1]

    if model_name in DiT_CONFIG.keys():
        config = DiT_CONFIG[model_name]
        config.update(**args.model_kwargs)
        config = EasyDict(config)
        model = DiT(args, config, **factor_kwargs)
        return model, config
    elif model_name in Aries_CONFIG.keys():
        config = Aries_CONFIG[model_name]
        config.update(**args.model_kwargs)
        config = EasyDict(config)
        cls = dict(Aries=Aries, AriesHF=AriesHF)[getattr(args, "model_structure", "Aries")]
        model = cls(args, config, **factor_kwargs)
        return model, config
    elif model_name in MMDiT_CONFIG.keys():
        config = MMDiT_CONFIG[model_name]
        config.update(**args.model_kwargs)
        config = EasyDict(config)
        model = Aries(args, config, **factor_kwargs)
        return model, config
    elif model_name in VQUNet_CONFIG.keys():
        config = VQUNet_CONFIG[model_name]
        config.update(**args.model_kwargs)
        config = EasyDict(config)
        model = VQUNet(args, config, logger, **factor_kwargs)
        return model, config
    elif model_name in TV2A_Aries_CONFIG.keys():
        config = TV2A_Aries_CONFIG[model_name]
        config.update(**args.model_kwargs)
        config = EasyDict(config)
        model = TV2A_Aries(args, config, **factor_kwargs)
        pretrained_model = kwargs.get("pretrained_ckpt", pretrained_model)
        if pretrained_model is None:
            pretrained_model = args.pretrained_ckpt
        if pretrained_model:
            state_dict = torch.load(pretrained_model, map_location="cpu")
            m, u = model.load_state_dict(state_dict["module"], strict=False)
            logger.info(
                f"loaded pretrained checkpoint:{args.pretrained_ckpt},\n missing keys: {m}\nunexpected keys: {u}"
            )
        return model, config
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
