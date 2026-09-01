# MeanFlow (iMF) sampler for the Leo two-time backbone.
#
# This is a NEW, self-contained module. It does NOT modify any existing file.
#
# It is identical to `Leo2Sampler`; its only purpose is that importing this module
# installs (on import) the `LeoModelMeanFlowHF` build dispatch and the MeanFlow sampling
# pipeline, so that the sampler entry can build the two-time iMF backbone and feed the
# second time input `r` at every denoising step.
#
# Usage (see jobs/examples/run_sample_leo_imf.sh):
#   --sampler leo2_meanflow_sampler.Leo2MeanFlowSampler --model-structure LeoModelMeanFlow
# (the entry turns `LeoModelMeanFlow` into `LeoModelMeanFlowHF`).

# Importing this installs the LeoModelMeanFlowHF dispatch + MeanFlow pipeline.
from ..models.diffusion import leo_meanflow_hf as _leo_meanflow_hf  # noqa: F401

from .leo2_sampler import Leo2Sampler


class Leo2MeanFlowSampler(Leo2Sampler):
    """Leo sampler for the iMF / MeanFlow backbone (two time inputs: t and r)."""
    pass
