from easydict import EasyDict

from .model import MLM, MASKGIT_CONFIG
from ...constants import PRETRAINED_LLM_PATH
from ...utils.torch_utils import load_state_dict


def build_model(args, pretrained_ckpt, logger, dtype=None, device=None):
    factor_kwargs = {'device': device, 'dtype': dtype}

    if args.model_name in MASKGIT_CONFIG.keys():
        model_config = MASKGIT_CONFIG[args.model_name]
        model_config.update(args.model_kwargs or {})
        model_config = EasyDict(model_config)

        model_ = MLM(args, model_config, **factor_kwargs)

        if pretrained_ckpt is None:
            pretrained_ckpt = args.pretrained_ckpt
        if pretrained_ckpt:
            if pretrained_ckpt in PRETRAINED_LLM_PATH:
                pretrained_ckpt = PRETRAINED_LLM_PATH[pretrained_ckpt]
            logger.info(f"loading pretrained checkpoint {pretrained_ckpt}")

            load_llm_to_mllm = args.get('load_llm_to_mllm', False)
            expand_keys = args.get('expand_keys', None)
            m, u = load_state_dict(model_, pretrained_ckpt, load_llm_to_mllm=load_llm_to_mllm, expand_keys=expand_keys)
            logger.info(
                f"loaded pretrained checkpoint,\n missing keys: {m}\nunexpected keys: {u}"
            )
        return model_, model_config

    else:
        raise NotImplementedError(f"model name not implemented for {args.model_name}")
