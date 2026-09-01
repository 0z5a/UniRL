import importlib
from loguru import logger

_pipeline_registry = {
    "token2image": "hymm.diffusion.pipelines.token2image_pipeline.Token2ImagePipeline",
    "cliptoken2image": "hymm.diffusion.pipelines.cliptoken2image_pipeline.ClipToken2ImagePipeline",
    "textvisual2audiodacvae": "hymm.diffusion.pipelines.textvisual2audio_pipeline.TextVisual2AudioPipeline",
    "text2image": "hymm.diffusion.pipelines.text2image_pipeline.Text2ImagePipeline",
}

def load_pipeline(name: str):
    # If name in registry, load the pipeline, else treat name as a module path
    if name in _pipeline_registry:
        module_path = _pipeline_registry[name]
        logger.warning("Loading pipeline with shorthand name will be deprecated. Please use full module path.")
    else:
        module_path = name

    assert "." in module_path, f"Invalid module path: {module_path} (should be like 'module.submodule.class')"
    module_name, class_name = module_path.rsplit(".", 1)
    module_spec = importlib.import_module(module_name)
    return getattr(module_spec, class_name)
