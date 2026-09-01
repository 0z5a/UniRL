from .pipelines import (
    MaskGITPipeline,
    Text2ImageMLMPipeline,
    PaintingMLMPipeline,
    EditingMLMPipeline,
    Text2ImageMARPipeline,
    Text2ImageMARTEPipeline,
    Text2ImageTransfusionPipeline,
    Text2SiglipTransfusionPipeline,
    Text2ImageTransfusionWithLogProbPipeline,
    DDPMPipeline,
)

from ..utils.helpers import default

_registered_pipelines = {
    "maskgit": MaskGITPipeline,
    "mlm": Text2ImageMLMPipeline,
    "mlm_inpainting": PaintingMLMPipeline,
    "mlm_editing": EditingMLMPipeline,
    "mar": Text2ImageMARPipeline,
    "mar_te": Text2ImageMARTEPipeline,
    "transfusion": Text2ImageTransfusionPipeline,
    "transfusion_with_logprob": Text2ImageTransfusionWithLogProbPipeline,
    "siglip_transfusion": Text2SiglipTransfusionPipeline,
}


def load_pipeline(args,
                  name,
                  rank=0,
                  model=None,
                  model_settings=None,
                  vae=None,
                  device=None,
                  progress_bar_config=None,
                  **kwargs,
                  ):

    # Only enable progress bar for rank 0
    progress_bar_config = default(progress_bar_config, {'leave': True, 'disable': rank != 0})

    pipeline = _registered_pipelines[name](vae=vae,
                                           model=model,
                                           model_settings=model_settings,
                                           progress_bar_config=progress_bar_config,
                                           args=args,
                                           **kwargs,
                                           )

    if not args.get("use_hf", False):
        pipeline = pipeline.to(device)

    return pipeline
