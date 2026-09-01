import importlib

MODEL_TO_ENGINES = {
    'multimodal': {
        'hunyuan-dense': 'hunyuan_multimodal_engine.HunyuanMultimodalEngine',
        'hunyuan-moe': 'hunyuan_multimodal_engine.HunyuanMultimodalEngine',
        'hunyuan3-moe': 'hunyuan_multimodal_engine.HunyuanMultimodalEngine',
        'qwen-vl-30b-a3b-instruct': 'hunyuan_multimodal_engine.HunyuanMultimodalEngine',
    },
    'diffusion': {
        'Aries': 'aries_engine.AriesEngine',
        'leo': 'leo_engine.LeoEngine',
    }
}


def find_engine(model_name):
    if "." not in model_name:
        raise ValueError(f"Engine name must be in the format 'module.EngineClassName', got '{model_name}'")

    # Determine engine name
    structure, name = model_name.rsplit(".", 1)
    if structure not in MODEL_TO_ENGINES:
        raise NotImplementedError(f"Model structure '{structure}' not found in MODEL_TO_ENGINES")
    for name_prefix in MODEL_TO_ENGINES[structure]:
        if name.startswith(name_prefix):
            engine_name = MODEL_TO_ENGINES[structure][name_prefix]
            break
    else:
        raise NotImplementedError(f"Model '{name}' not found in MODEL_TO_ENGINES for structure '{structure}'")

    # Get engine class
    module_name, class_name = engine_name.rsplit(".", 1)
    module = importlib.import_module(f"hymm.engines.{module_name}")
    return getattr(module, class_name)
